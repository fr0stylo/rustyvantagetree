#![feature(test, portable_simd)]

use std::{
    collections::BTreeSet,
    fmt::Debug,
    fs::OpenOptions,
    io::{BufRead, BufReader, Error},
    simd::u8x8,
    sync::{Arc, Mutex},
};

use anyerror::AnyError;

const CHUNK_SIZE: usize = 8;
const LEAF_CAP: usize = 50;

fn to_bit_string(data: &Vec<u8>) -> String {
    data.iter()
        .map(|byte| format!("{:08b}", byte))
        .collect::<Vec<String>>()
        .join("")
}

fn hamming_distance(x: &Vec<u8>, y: &Vec<u8>) -> usize {
    let len = x.len();
    let mut count = 0;

    // Process 16 bytes at a time
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

fn hamming_distance_list(x: &Vec<u8>, y: &Vec<u8>) -> usize {
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
        i: &Vec<u8>,
        radius: usize,
        k: usize,
        results: Arc<Mutex<Vec<(usize, Vec<u8>)>>>,
    ) -> Result<(), AnyError>;

    fn add(&mut self, i: &Vec<u8>) -> Result<usize, AnyError>;
    fn size(&self) -> usize;
}

#[derive(Debug)]
pub struct VPNode {
    vantage_point: Vec<u8>,
    threshhold: Option<usize>,
    values: BTreeSet<Vec<u8>>,
    near_nodes: Option<Box<VPNode>>,
    far_nodes: Option<Box<VPNode>>,
}

impl VPNode {
    pub fn new(val: &Vec<u8>) -> Self {
        Self {
            vantage_point: val.clone(),
            threshhold: None,
            values: BTreeSet::new(),
            near_nodes: None,
            far_nodes: None,
        }
    }

    fn create_leaf(&mut self, i: &Vec<u8>, dist: usize) {
        if self.values.len() < LEAF_CAP {
            self.values.insert(i.clone());
        } else {
            let vantage_point = self.vantage_point.clone();
            let nodes: Vec<(usize, Vec<u8>)> = self
                .values
                .iter()
                .map(|x| (hamming_distance(x, &vantage_point), x.clone()))
                .collect();
            let mut distances: Vec<usize> = nodes.iter().map(|&(x, _)| x).collect();
            distances.sort_by_key(|&x| x);
            let threshold = median_sorted_vec(&distances);
            self.threshhold = Some(threshold);

            self.far_nodes = self.create_node(|x| hamming_distance(x, &vantage_point) > threshold);
            self.near_nodes =
                self.create_node(|x| hamming_distance(x, &vantage_point) <= threshold);
            self.values.clear();
        }
    }

    fn create_node<F>(&mut self, f: F) -> Option<Box<VPNode>>
    where
        F: Fn(&Vec<u8>) -> bool,
    {
        let mut values: Vec<Vec<u8>> = self
            .values
            .iter()
            .filter(|&x| f(x))
            .map(|x| x.clone())
            .collect();

        values.sort_by_key(|x| hamming_distance(x, &self.vantage_point));

        let root = median_sorted_vec(&values);

        let mut node = VPNode::new(&root);
        values.iter().for_each(|x| {
            let _ = node.add(&x);
        });

        Some(Box::new(node))
    }
}

impl TreeNode for VPNode {
    fn search(
        &self,
        i: &Vec<u8>,
        radius: usize,
        k: usize,
        results: Arc<Mutex<Vec<(usize, Vec<u8>)>>>,
    ) -> Result<(), AnyError> {
        let dist = hamming_distance(&self.vantage_point, &i);
        if dist <= radius {
            results.lock().unwrap().push((dist, i.clone()));
        }

        if self.near_nodes.is_none() && self.far_nodes.is_none() {
            let res = self
                .values
                .iter()
                .map(|x| (hamming_distance(x, &i), x.clone()))
                .filter(|&(x, _)| x <= radius)
                .collect::<Vec<(usize, Vec<u8>)>>();
            results.lock().unwrap().extend(res);
        } else {
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

    fn add(&mut self, i: &Vec<u8>) -> Result<usize, AnyError> {
        let dist = hamming_distance(&self.vantage_point, &i);
        if dist == 0 {
            return Ok(dist);
        }

        if self.near_nodes.is_none() && self.far_nodes.is_none() {
            self.create_leaf(i, dist);
        } else if dist > self.threshhold.unwrap() {
            return self.far_nodes.as_mut().unwrap().add(i);
        } else {
            return self.near_nodes.as_mut().unwrap().add(i);
        }

        Ok(dist)
    }

    fn size(&self) -> usize {
        if !self.near_nodes.is_none() && !self.far_nodes.is_none() {
            1 + self.far_nodes.as_ref().unwrap().size() + self.near_nodes.as_ref().unwrap().size()
        } else {
            self.values.len()
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
        val: &Vec<u8>,
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
                    .unwrap_or_else(|| Arc::new(Mutex::new(VPNode::new(&x.to_vec()))));

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

    fn size(&self) -> usize {
        self.tree.clone().unwrap().lock().unwrap().size()
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
