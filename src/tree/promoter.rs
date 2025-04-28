use std::sync::Arc;

use super::{DistanceMetric, TreeNode};

pub trait TreeNodePromoter<V, T> {
    fn promote(
        &self,
        _distance: &Arc<dyn DistanceMetric<[V], T>>,
    ) -> Option<Box<dyn TreeNode<V, T>>> {
        None
    }

    fn need_promotion(&self) -> bool {
        false
    }
}
