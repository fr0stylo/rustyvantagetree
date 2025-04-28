use std::sync::{Arc, Mutex};

use anyerror::AnyError;

use super::{
    promoter::TreeNodePromoter,
    result_set::{ResultEntry, ResultSet},
    BranchPruner, DistanceMetric, TreeMetrics, TreeNode, TreeNodeMetrics,
};

#[derive(Debug)]
pub struct VPNode<V, T> {
    vantage_point: Option<Vec<V>>,
    threshold: Option<T>,
    near_nodes: Option<Box<dyn TreeNode<V, T>>>,
    far_nodes: Option<Box<dyn TreeNode<V, T>>>,
}

impl VPNode<u8, usize> {
    pub fn new(
        vp: Vec<u8>,
        threshold: usize,
        nearn: Box<dyn TreeNode<u8, usize>>,
        farn: Box<dyn TreeNode<u8, usize>>,
    ) -> Self {
        Self {
            vantage_point: Some(vp),
            threshold: Some(threshold),
            near_nodes: Some(nearn),
            far_nodes: Some(farn),
        }
    }

    pub fn default() -> Self {
        Self {
            vantage_point: None,
            threshold: None,
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
        results: Arc<Mutex<ResultSet<ResultEntry>>>,
    ) -> Result<(), AnyError> {
        let dist = distance.distance(self.vantage_point.clone().unwrap().as_slice(), i);
        if dist <= radius {
            results
                .lock()
                .expect("Results were locked")
                .push(ResultEntry::from_tuple((dist, i.to_vec())));
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
        let mut metrics = TreeMetrics::node();

        if let Some(near_nodes) = &self.near_nodes {
            metrics += near_nodes.size();
        }
        if let Some(far_nodes) = &self.far_nodes {
            metrics += far_nodes.size();
        }

        metrics.update_depth();

        return metrics;
    }
}
