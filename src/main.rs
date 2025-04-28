#![feature(test, portable_simd, allocator_api)]

pub mod tree;

use std::{
    fs::OpenOptions,
    io::{BufRead, BufReader, Error},
    sync::{Arc, Mutex},
};

use anyerror::AnyError;

use tree::{
    distance::{DistanceMetric, HammingDistance},
    leaf::VPLeaf,
    metrics::{TreeMetrics, TreeNodeMetrics},
    promoter::TreeNodePromoter,
    result_set::{ResultEntry, ResultSet},
    tree_node::TreeNode,
    treeshaker::{AproximateShaker, BranchPruner, ExactShaker},
};

const CHUNK_SIZE: usize = 8;
const LEAF_CAP: usize = 512;

// fn to_bit_string(data: &[u8]) -> String {
//     data.iter()
//         .map(|byte| format!("{:08b}", byte))
//         .collect::<Vec<String>>()
//         .join("")
// }

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
    ) -> Result<Vec<ResultEntry>, AnyError> {
        if val.len() != self.size {
            return Err(AnyError::error(format!(
                "input vec (len: {}) does not match size ({})",
                val.len(),
                self.size
            )));
        }
        let results = Arc::new(Mutex::new(ResultSet::new(k)));
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

        let values: Vec<ResultEntry> = results.lock().unwrap().collect();
        Ok(values)
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
