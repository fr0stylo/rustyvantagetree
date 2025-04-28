#![feature(test, portable_simd)]

pub mod distance;
pub mod treeshaker;

use std::{
    cmp::{max, min},
    fmt::Debug,
    fs::OpenOptions,
    io::{BufRead, BufReader, Error},
    marker::PhantomData,
    ops::{AddAssign, Deref},
    sync::{Arc, Mutex},
    usize,
};

use anyerror::AnyError;
use distance::{DistanceMetric, HammingDistance};

use treeshaker::*;

const CHUNK_SIZE: usize = 8;
const LEAF_CAP: usize = 20;

// fn to_bit_string(data: &[u8]) -> String {
//     data.iter()
//         .map(|byte| format!("{:08b}", byte))
//         .collect::<Vec<String>>()
//         .join("")
// }

fn median_sorted_vec<T>(vec: &Vec<T>) -> T
where
    T: Clone,
{
    let len = vec.len();
    if len == 0 {
        panic!("Empty vector has no median");
    }

    if len % 2 == 1 {
        vec[len / 2].clone()
    } else {
        vec[len / 2 - 1].clone()
    }
}

pub trait TreeNode<V, T>: Debug + Sync + TreeNodeMetrics + TreeNodePromoter<V, T> {
    fn search(
        &self,
        pruner: &Arc<dyn BranchPruner<T>>,
        distance: &Arc<dyn DistanceMetric<[V], T>>,
        i: &[V],
        radius: T,
        k: usize,
        results: Arc<Mutex<Vec<(T, Vec<V>)>>>,
    ) -> Result<(), AnyError>;

    fn add(&mut self, distance: &Arc<dyn DistanceMetric<[V], T>>, i: &[V]) -> Result<T, AnyError>;
    // fn size(&self) -> (usize, usize, usize);
}

#[derive(Debug)]
pub struct TreeMetrics {
    nodes: usize,
    leaves: usize,
    elements: usize,
    max_depth: usize,
    min_depth: usize,
    steps: usize,
    avg_depth: f32,
}

impl TreeMetrics {
    fn new() -> Self {
        TreeMetrics {
            nodes: 0,
            leaves: 0,
            elements: 0,
            max_depth: 0,
            min_depth: usize::MAX,
            steps: 0,
            avg_depth: 0.0,
        }
    }
    fn leaf(size: usize) -> Self {
        TreeMetrics {
            nodes: 0,
            leaves: 1,
            elements: size,
            max_depth: 0,
            min_depth: 1,
            steps: 0,
            avg_depth: 0.0,
        }
    }
}

impl AddAssign for TreeMetrics {
    fn add_assign(&mut self, rhs: Self) {
        self.nodes += rhs.nodes;
        self.leaves += rhs.leaves;
        self.elements += rhs.elements;
        self.steps += rhs.steps + 1;

        self.max_depth = max(self.max_depth, rhs.max_depth);
        self.min_depth = min(self.min_depth, rhs.min_depth);
        self.avg_depth = self.steps as f32 / self.leaves as f32;
    }
}

pub trait TreeNodeMetrics {
    fn size(&self) -> TreeMetrics;
}
pub trait TreeNodePromoter<V, T> {
    fn promote(
        &self,
        distance: &Arc<dyn DistanceMetric<[V], T>>,
    ) -> Option<Box<dyn TreeNode<V, T>>> {
        None
    }

    fn need_promotion(&self) -> bool {
        false
    }
}

#[derive(Debug)]
pub struct VPLeaf<V, T> {
    values: Vec<Vec<V>>,
    _threshold: Option<T>,
}

impl<V, T> VPLeaf<V, T> {
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
            _threshold: None,
        }
    }

    fn extend(&mut self, i: Vec<Vec<V>>) {
        self.values.extend(i);
    }
}

impl VPLeaf<u8, usize> {
    fn make_leaf(&self, f: Vec<Vec<u8>>) -> Option<Box<dyn TreeNode<u8, usize>>> {
        let mut node = VPLeaf::<u8, usize>::new();

        node.extend(f);

        return Some(Box::new(node));
    }
}

impl TreeNodePromoter<u8, usize> for VPLeaf<u8, usize> {
    fn promote(
        &self,
        distance: &Arc<dyn DistanceMetric<[u8], usize>>,
    ) -> Option<Box<dyn TreeNode<u8, usize>>> {
        if self.values.len() < 2 {
            return None; // Can't promote a leaf with fewer than 2 values
        }

        let mut node = VPNode::<u8, usize>::new();

        let mut vantage_points: Vec<(usize, Vec<(usize, Vec<u8>)>, Vec<u8>)> = self
            .values
            .iter()
            .map(|x| {
                let distances: Vec<(usize, Vec<u8>)> = self
                    .values
                    .iter()
                    .filter(|&y| !x.eq(y))
                    .map(|y| (distance.distance(x, y), y.clone()))
                    .collect();

                let avg_distance: usize =
                    distances.iter().fold(0, |acc, &(x, _)| acc + x) / distances.len();

                (avg_distance, distances, x.clone())
            })
            .collect();

        vantage_points.sort_by_key(|&(x, _, _)| x);

        let (_avg_distance, mut distances, vantage_point) = median_sorted_vec(&vantage_points);
        node.vantage_point = Some(vantage_point.clone());

        distances.sort_by_key(|&(x, _)| x);
        let (threshold, _) = median_sorted_vec(&distances);
        node.threshold = Some(threshold);

        node.far_nodes = self.make_leaf(
            distances
                .iter()
                .filter(|(x, _)| x.clone() > threshold && x.clone() != 0)
                .map(|(_, x)| x.clone())
                .collect(),
        );
        node.near_nodes = self.make_leaf(
            distances
                .iter()
                .filter(|(x, _)| x.clone() <= threshold && x.clone() != 0)
                .map(|(_, x)| x.clone())
                .collect(),
        );

        Some(Box::new(node))
    }

    fn need_promotion(&self) -> bool {
        self.values.len() >= LEAF_CAP
    }
}

impl<V, T> TreeNodeMetrics for VPLeaf<V, T> {
    fn size(&self) -> TreeMetrics {
        return TreeMetrics::leaf(self.values.len());
    }
}

impl TreeNode<u8, usize> for VPLeaf<u8, usize> {
    fn search(
        &self,
        _pruner: &Arc<dyn BranchPruner<usize>>,
        distance: &Arc<dyn DistanceMetric<[u8], usize>>,
        i: &[u8],
        radius: usize,
        _k: usize,
        results: Arc<Mutex<Vec<(usize, Vec<u8>)>>>,
    ) -> Result<(), AnyError> {
        let res = self
            .values
            .iter()
            .map(|x| (distance.distance(x, &i), x.clone()))
            .filter(|&(x, _)| x <= radius)
            .collect::<Vec<(usize, Vec<u8>)>>();
        results.lock().expect("Results were locked").extend(res);
        Ok(())
    }

    fn add(
        &mut self,
        _distance: &Arc<dyn DistanceMetric<[u8], usize>>,
        i: &[u8],
    ) -> Result<usize, AnyError> {
        // if self.values.len() <= LEAF_CAP {
        let vec = i.to_vec();
        if !self.values.contains(&vec) {
            self.values.push(vec);
        }
        // }

        Ok(1)
    }
}

#[derive(Debug)]
pub struct VPNode<V, T> {
    vantage_point: Option<Vec<V>>,
    threshold: Option<T>,
    values: Vec<Vec<V>>,
    near_nodes: Option<Box<dyn TreeNode<V, T>>>,
    far_nodes: Option<Box<dyn TreeNode<V, T>>>,
}

impl VPNode<u8, usize> {
    pub fn new() -> Self {
        Self {
            vantage_point: None,
            threshold: None,
            values: Vec::new(),
            near_nodes: None,
            far_nodes: None,
        }
    }
}

impl<V, T> TreeNodePromoter<V, T> for VPNode<V, T> {}

impl TreeNode<u8, usize> for VPNode<u8, usize> {
    fn search(
        &self,
        pruner: &Arc<dyn BranchPruner<usize>>,
        distance: &Arc<dyn DistanceMetric<[u8], usize>>,
        i: &[u8],
        radius: usize,
        k: usize,
        results: Arc<Mutex<Vec<(usize, Vec<u8>)>>>,
    ) -> Result<(), AnyError> {
        if self.near_nodes.is_none() && self.far_nodes.is_none() {
            let res = self
                .values
                .iter()
                .map(|x| (distance.distance(x, &i), x.clone()))
                .filter(|&(x, _)| x <= radius)
                .collect::<Vec<(usize, Vec<u8>)>>();
            results.lock().expect("Results were locked").extend(res);
        } else if let Some(vantage_point) = &self.vantage_point {
            let dist = distance.distance(vantage_point, i);
            if dist <= radius {
                results
                    .lock()
                    .expect("Results were locked")
                    .push((dist, i.to_vec()));
            }

            let (near_shake, far_shake) = pruner.prune(dist, radius, self.threshold.unwrap());

            rayon::join(
                || {
                    if near_shake {
                        if let Some(near_nodes) = &self.near_nodes {
                            let _ = near_nodes.search(
                                &pruner,
                                &distance,
                                i,
                                radius,
                                k,
                                Arc::clone(&results),
                            );
                        }
                    }
                },
                || {
                    if far_shake {
                        if let Some(far_nodes) = &self.far_nodes {
                            let _ = far_nodes.search(
                                &pruner,
                                &distance,
                                i,
                                radius,
                                k,
                                Arc::clone(&results),
                            );
                        }
                    }
                },
            );
        }

        return Ok(());
    }

    fn add(
        &mut self,
        distance: &Arc<dyn DistanceMetric<[u8], usize>>,
        i: &[u8],
    ) -> Result<usize, AnyError> {
        let dist = distance.distance(&self.vantage_point.clone().unwrap(), &i);
        if dist == 0 {
            return Ok(dist);
        }

        if dist > self.threshold.unwrap() {
            if let Some(far_nodes) = &self.far_nodes {
                if far_nodes.need_promotion() {
                    if let Some(promoted) = far_nodes.promote(distance) {
                        self.far_nodes = Some(promoted)
                    }
                }
            }
            return self.far_nodes.as_mut().unwrap().add(&distance, i);
        } else {
            if let Some(near_nodes) = &self.near_nodes {
                if near_nodes.need_promotion() {
                    if let Some(promoted) = near_nodes.promote(distance) {
                        self.near_nodes = Some(promoted)
                    }
                }
            }
            return self.near_nodes.as_mut().unwrap().add(&distance, i);
        }
    }
}

impl<V, T> TreeNodeMetrics for VPNode<V, T> {
    fn size(&self) -> TreeMetrics {
        let mut metrics = TreeMetrics::new();

        if let Some(_) = &self.vantage_point {
            metrics.nodes += 1;
            if let Some(near_nodes) = &self.near_nodes {
                metrics += near_nodes.size();
            }
            if let Some(far_nodes) = &self.far_nodes {
                metrics += far_nodes.size();
            }
        } else {
            metrics.leaves = 1;
            metrics.min_depth = 1;
            metrics.elements = self.values.len() + 1;
        }

        metrics.max_depth += 1;
        metrics.min_depth = metrics.min_depth.saturating_add(1);

        return metrics;
    }
}

#[derive(Debug)]
pub struct VantagePointTree<'a, V, T> {
    size: T,
    tree: Box<dyn TreeNode<V, T>>,
    pruner: &'a Arc<dyn BranchPruner<T>>,
    distance: &'a Arc<dyn DistanceMetric<[V], T>>,
}

impl<'a> VantagePointTree<'a, u8, usize> {
    pub fn new(
        size: usize,
        pruner: &'a Arc<dyn BranchPruner<usize>>,
        distancer: &'a Arc<dyn DistanceMetric<[u8], usize>>,
    ) -> Self {
        VantagePointTree {
            size: size,
            tree: Box::new(VPLeaf::<u8, usize>::new()),
            pruner: pruner,
            distance: distancer,
        }
    }

    pub fn search(
        &self,
        val: &[u8],
        radius: usize,
        k: usize,
    ) -> Result<Vec<(usize, Vec<u8>)>, AnyError> {
        if val.len() != self.size {
            return Err(AnyError::error(format!(
                "input vec (len: {}) does not match size ({})",
                val.len(),
                self.size
            )));
        }
        let results = Arc::new(Mutex::new(Vec::new()));
        let _ = &self
            .tree
            .search(
                &self.pruner,
                &self.distance,
                val,
                radius,
                50,
                Arc::clone(&results),
            )
            .unwrap();
        results.lock().unwrap().sort_by_key(|&(x, _)| x);
        let top: Vec<(usize, Vec<u8>)> = results
            .lock()
            .unwrap()
            .iter()
            .take(k)
            .map(|x| x.clone())
            .collect();
        Ok(top)
    }

    pub fn add(&mut self, item: Vec<u8>) -> Result<Vec<usize>, AnyError> {
        let res = item
            .chunks(self.size)
            .map(|x| {
                assert!(
                    x.len() == self.size,
                    "All chunks must be {} multiples was {}",
                    self.size,
                    x.len()
                );

                if self.tree.need_promotion() {
                    if let Some(tree) = self.tree.promote(&self.distance) {
                        self.tree = tree;
                    }
                }

                return self.tree.as_mut().add(&self.distance, &x.to_vec());
            })
            .map(|x| x.unwrap())
            .collect();
        Ok(res)
    }

    fn size(&self) -> TreeMetrics {
        return self.tree.size();
    }
}

fn seed_tree(tree: &mut VantagePointTree<u8, usize>) -> Result<(), Error> {
    let values = OpenOptions::new().read(true).open("./file.vals")?;
    let mut buf = String::new();
    let mut reader = BufReader::new(values);

    loop {
        let len = reader.read_line(&mut buf)?;
        if len == 0 {
            break;
        }

        let _ = tree.add(buf.trim().as_bytes().to_vec()).unwrap();

        buf.clear();
    }

    Ok(())
}

fn main() -> Result<(), Error> {
    let pruner: Arc<dyn BranchPruner<usize>> = Arc::new(&ExactShaker);
    let hamming: Arc<dyn DistanceMetric<[u8], usize>> = Arc::new(HammingDistance::new(8));

    let mut tree = VantagePointTree::<u8, usize>::new(CHUNK_SIZE, &pruner, &hamming);
    seed_tree(&mut tree)?;
    println!("Construction completed");
    // println!("{:?}", tree);
    let term = "34ec86d2".as_bytes().to_vec();

    println!("tree size {:?}", tree.size());
    println!("result {:?}", tree.search(&term, 10, 10));
    let pruner: Arc<dyn BranchPruner<usize>> = Arc::new(&AproximateShaker);

    tree.pruner = &pruner;
    println!("result {:?}", tree.search(&term, 10, 10));

    Ok(())
}

extern crate test;

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;

    #[bench]
    fn bench_tree_search_exact(b: &mut Bencher) {
        let pruner: Arc<dyn BranchPruner<usize>> = Arc::new(&ExactShaker);
        let hamming: Arc<dyn DistanceMetric<[u8], usize>> = Arc::new(HammingDistance::new(8));

        let mut tree = VantagePointTree::new(CHUNK_SIZE, &pruner, &hamming);
        seed_tree(&mut tree).unwrap();

        let term = "34ec86d2".as_bytes().to_vec();

        b.iter(|| tree.search(&term, 5, 10));
    }

    #[bench]
    fn bench_tree_search_approximate(b: &mut Bencher) {
        let pruner: Arc<dyn BranchPruner<usize>> = Arc::new(&AproximateShaker);
        let hamming: Arc<dyn DistanceMetric<[u8], usize>> = Arc::new(HammingDistance::new(8));

        let mut tree = VantagePointTree::new(CHUNK_SIZE, &pruner, &hamming);
        seed_tree(&mut tree).unwrap();

        let term = "34ec86d2".as_bytes().to_vec();

        b.iter(|| tree.search(&term, 5, 10));
    }
}
