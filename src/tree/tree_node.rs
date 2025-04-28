use std::{
    fmt::Debug,
    sync::{Arc, Mutex},
};

use anyerror::AnyError;

use crate::TreeNodePromoter;

use super::{
    result_set::{ResultEntry, ResultSet},
    BranchPruner, DistanceMetric, TreeNodeMetrics,
};

pub trait TreeNode<V, T>: Debug + Sync + TreeNodeMetrics + TreeNodePromoter<V, T> {
    fn search(
        &self,
        pruner: &Arc<dyn BranchPruner<T>>,
        distance: &Arc<dyn DistanceMetric<[V], T>>,
        i: &[V],
        radius: T,
        k: usize,
        results: Arc<Mutex<ResultSet<ResultEntry>>>,
    ) -> Result<(), AnyError>;

    fn add(&mut self, distance: &Arc<dyn DistanceMetric<[V], T>>, i: &[V]) -> Result<T, AnyError>;
    // fn size(&self) -> (usize, usize, usize);
}
