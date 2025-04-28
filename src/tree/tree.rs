use std::sync::{Arc, Mutex};

use anyerror::AnyError;

use super::{
    leaf::VPLeaf,
    result_set::{ResultEntry, ResultSet},
    BranchPruner, DistanceMetric, TreeMetrics, TreeNode,
};

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

        let mut values: Vec<ResultEntry> = results.lock().unwrap().collect();
        values.sort_by_key(|x| x.dist());
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

    pub fn size(&self) -> TreeMetrics {
        return self.tree.size();
    }
}
