#![feature(test, portable_simd)]

use std::{
    fmt::Debug,
    fs::OpenOptions,
    io::{BufRead, BufReader, Error},
    simd::u8x8,
    sync::{Arc, Mutex},
};

use anyerror::AnyError;

const CHUNK_SIZE: usize = 8;
const LEAF_CAP: usize = 100;

fn to_bit_string(data: &[u8]) -> String {
    data.iter()
        .map(|byte| format!("{:08b}", byte))
        .collect::<Vec<String>>()
        .join("")
}

fn hamming_distance(x: &[u8], y: &[u8]) -> usize {
    let len = x.len();
    let mut count = 0;

    let chunks = len / CHUNK_SIZE;
    for i in 0..chunks {
        let start = i * CHUNK_SIZE;

        let x_vec = u8x8::from_slice(&x[start..start + CHUNK_SIZE]);
        let y_vec = u8x8::from_slice(&y[start..start + CHUNK_SIZE]);

        let xor_result = x_vec ^ y_vec;

        for j in 0..CHUNK_SIZE {
            count += xor_result[j].count_ones() as usize;
        }
    }

    // Handle remaining bytes
    let remainder_start = chunks * CHUNK_SIZE;
    for i in remainder_start..len {
        count += (x[i] ^ y[i]).count_ones() as usize;
    }

    count
}

fn hamming_distance_list(x: &[u8], y: &[u8]) -> usize {
    x.iter()
        .zip(y.iter())
        .fold(0, |acc, (x, y)| acc + (x ^ y).count_ones() as usize)
}

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

pub trait TreeNode: Debug {
    fn search(
        &self,
        i: &[u8],
        radius: usize,
        k: usize,
        results: Arc<Mutex<Vec<(usize, Vec<u8>)>>>,
    ) -> Result<(), AnyError>;

    fn add(&mut self, i: &[u8]) -> Result<usize, AnyError>;
    fn size(&self) -> (usize, usize, usize);
}

#[derive(Debug)]
pub struct VPNode {
    vantage_point: Option<Vec<u8>>,
    threshhold: Option<usize>,
    values: Vec<Vec<u8>>,
    near_nodes: Option<Box<VPNode>>,
    far_nodes: Option<Box<VPNode>>,
}

impl VPNode {
    pub fn new() -> Self {
        Self {
            vantage_point: None,
            threshhold: None,
            values: Vec::new(),
            near_nodes: None,
            far_nodes: None,
        }
    }

    fn create_leaf(&mut self, i: &[u8]) {
        if self.values.len() < LEAF_CAP {
            let vec = i.to_vec();
            if !self.values.contains(&vec) {
                self.values.push(vec);
            }
        } else {
            let mut vantage_points: Vec<(usize, Vec<(usize, Vec<u8>)>, Vec<u8>)> = self
                .values
                .iter()
                .map(|x| {
                    let distances: Vec<(usize, Vec<u8>)> = self
                        .values
                        .iter()
                        .filter(|&y| !x.eq(y))
                        .map(|y| (hamming_distance(x, y), y.clone()))
                        .collect();

                    let avg_distance: usize =
                        distances.iter().fold(0, |acc, &(x, _)| acc + x) / distances.len();

                    (avg_distance, distances, x.clone())
                })
                .collect();

            vantage_points.sort_by_key(|&(x, _, _)| x);

            let (_avg_distance, mut distances, vantage_point) = median_sorted_vec(&vantage_points);
            self.vantage_point = Some(vantage_point.clone());

            distances.sort_by_key(|&(x, _)| x);
            let (threshold, _) = median_sorted_vec(&distances);
            self.threshhold = Some(threshold);

            self.far_nodes = self.create_node(|x| hamming_distance(x, &vantage_point) > threshold);
            self.near_nodes =
                self.create_node(|x| hamming_distance(x, &vantage_point) <= threshold);
            self.values.clear();
        }
    }

    fn create_node<F>(&mut self, f: F) -> Option<Box<VPNode>>
    where
        F: Fn(&[u8]) -> bool,
    {
        if let Some(_) = &self.vantage_point {
            let values: Vec<Vec<u8>> = self
                .values
                .iter()
                .filter(|&x| f(x))
                .map(|x| x.clone())
                .collect();

            let mut node = VPNode::new();
            values.iter().for_each(|x| {
                let _ = node.add(&x);
            });

            return Some(Box::new(node));
        }
        None
    }
}

impl TreeNode for VPNode {
    fn search(
        &self,
        i: &[u8],
        radius: usize,
        k: usize,
        results: Arc<Mutex<Vec<(usize, Vec<u8>)>>>,
    ) -> Result<(), AnyError> {
        if self.near_nodes.is_none() && self.far_nodes.is_none() {
            let res = self
                .values
                .iter()
                .map(|x| (hamming_distance(x, &i), x.clone()))
                .filter(|&(x, _)| x <= radius)
                .collect::<Vec<(usize, Vec<u8>)>>();
            results.lock().expect("Results were locked").extend(res);
        } else if let Some(vantage_point) = &self.vantage_point {
            let dist = hamming_distance(vantage_point, i);
            if dist <= radius {
                results
                    .lock()
                    .expect("Results were locked")
                    .push((dist, i.to_vec()));
            }

            rayon::join(
                || {
                    if dist.saturating_sub(radius) <= self.threshhold.unwrap() {
                        if let Some(near_nodes) = &self.near_nodes {
                            let _ = near_nodes.search(i, radius, k, Arc::clone(&results));
                        }
                    }
                },
                || {
                    if dist.saturating_add(radius) > self.threshhold.unwrap() {
                        if let Some(far_nodes) = &self.far_nodes {
                            let _ = far_nodes.search(i, radius, k, Arc::clone(&results));
                        }
                    }
                },
            );
        }

        return Ok(());
    }

    fn add(&mut self, i: &[u8]) -> Result<usize, AnyError> {
        if let Some(vantage_point) = &self.vantage_point {
            let dist = hamming_distance(&vantage_point, &i);
            if dist == 0 {
                return Ok(dist);
            }

            if dist > self.threshhold.unwrap() {
                return self.far_nodes.as_mut().unwrap().add(i);
            } else {
                return self.near_nodes.as_mut().unwrap().add(i);
            }
        } else {
            self.create_leaf(i);
        }

        Ok(0)
    }

    fn size(&self) -> (usize, usize, usize) {
        if let Some(_) = &self.vantage_point {
            let mut counter = (0, 0, 0);
            if let Some(near_nodes) = &self.near_nodes {
                let (nodes, leaves, elements) = near_nodes.size();
                counter.0 += nodes + 1;
                counter.1 += leaves;
                counter.2 += elements + 1;
            }
            if let Some(far_nodes) = &self.far_nodes {
                let (nodes, leaves, elements) = far_nodes.size();
                counter.0 += nodes + 1;
                counter.1 += leaves;
                counter.2 += elements + 1;
            }

            counter
        } else {
            (0, 1, self.values.len())
        }
    }
}

#[derive(Debug)]
pub struct VantagePointTree {
    size: usize,
    tree: Option<Arc<Mutex<dyn TreeNode>>>,
}

impl VantagePointTree {
    pub fn new(size: usize) -> Self {
        VantagePointTree {
            size: size,
            tree: None,
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
        match self.tree.clone() {
            Some(tree) => {
                let results = Arc::new(Mutex::new(Vec::new()));
                let _ = tree
                    .lock()
                    .unwrap()
                    .search(val, radius, 50, Arc::clone(&results))
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
            None => Err(AnyError::error("Tree has not been created")),
        }
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

                let root = self
                    .tree
                    .clone()
                    .unwrap_or_else(|| Arc::new(Mutex::new(VPNode::new())));

                match root.clone().lock().unwrap().add(&x.to_vec()) {
                    Ok(x) => {
                        self.tree = Some(root.clone());
                        Ok(x)
                    }
                    Err(e) => return Err(e),
                }
            })
            .map(|x| x.unwrap())
            .collect();
        Ok(res)
    }

    fn size(&self) -> (usize, usize, usize) {
        if let Some(tree) = &self.tree {
            if let Ok(tree) = tree.lock() {
                return tree.size();
            }
        }

        (0, 0, 0)
    }
}

fn seed_tree(tree: &mut VantagePointTree) -> Result<(), Error> {
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
    let mut tree = VantagePointTree::new(CHUNK_SIZE);
    seed_tree(&mut tree)?;

    // println!("{:?}", tree);
    let term = "34ec86d2".as_bytes().to_vec();

    println!("tree size {:?}", tree.size());
    println!("result {:?}", tree.search(&term, 10, 10));

    Ok(())
}

extern crate test;

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;

    #[bench]
    fn bench_tree_search(b: &mut Bencher) {
        let mut tree = VantagePointTree::new(CHUNK_SIZE);
        seed_tree(&mut tree).unwrap();

        let term = "34ec86d2".as_bytes().to_vec();

        b.iter(|| tree.search(&term, 5, 10));
    }

    #[bench]
    fn bench_hamming_distance_list(b: &mut Bencher) {
        let term1 = "34ec86d2".as_bytes().to_vec();
        let term2 = "35ef86d2".as_bytes().to_vec();

        b.iter(|| hamming_distance_list(&term1, &term2));
    }

    #[bench]
    fn bench_hamming_distance(b: &mut Bencher) {
        let term1 = "34ec86d2".as_bytes().to_vec();
        let term2 = "35ef86d2".as_bytes().to_vec();

        b.iter(|| hamming_distance(&term1, &term2));
    }
}
