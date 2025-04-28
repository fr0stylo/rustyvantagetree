#![feature(test, portable_simd, allocator_api)]

pub mod tree;

use std::{
    fs::OpenOptions,
    io::{BufRead, BufReader, Error},
    sync::Arc,
};

use tree::{
    distance::{DistanceMetric, HammingDistance},
    treeshaker::{BranchPruner, ExactShaker},
    VantagePointTree,
};

const CHUNK_SIZE: usize = 8;
const LEAF_CAP: usize = 512;

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
    let term = "03acb7ad".as_bytes().to_vec();

    println!("tree size {:?}", tree.size());
    println!("result {:?}", tree.search(&term, 10, 10));

    Ok(())
}

extern crate test;

#[cfg(test)]
mod tests {
    use crate::tree::AproximateShaker;

    use super::*;
    use test::Bencher;

    #[bench]
    fn bench_tree_search_exact(b: &mut Bencher) {
        let pruner: Arc<dyn BranchPruner<usize>> = Arc::new(&ExactShaker);
        let hamming: Arc<dyn DistanceMetric<[u8], usize>> = Arc::new(HammingDistance::new(8));

        let mut tree = VantagePointTree::new(CHUNK_SIZE, &pruner, &hamming);
        seed_tree(&mut tree).unwrap();

        let term = "03acb7ad".as_bytes().to_vec();

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
