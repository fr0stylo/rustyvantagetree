use std::sync::{Arc, Mutex};

use anyerror::AnyError;

use super::{
    median_sorted_vec,
    promoter::TreeNodePromoter,
    result_set::{ResultEntry, ResultSet},
    BranchPruner, DistanceMetric, TreeMetrics, TreeNode, TreeNodeMetrics, VPNode,
};

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
        distances.sort_by_key(|&(x, _)| x);
        let (threshold, _) = median_sorted_vec(&distances);

        let far = self.make_leaf(
            distances
                .iter()
                .filter(|(x, _)| x.clone() > threshold && x.clone() != 0)
                .map(|(_, x)| x.clone())
                .collect(),
        );
        let near = self.make_leaf(
            distances
                .iter()
                .filter(|(x, _)| x.clone() <= threshold && x.clone() != 0)
                .map(|(_, x)| x.clone())
                .collect(),
        );

        let node = VPNode::<u8, usize>::new(vantage_point, threshold, near.unwrap(), far.unwrap());

        Some(Box::new(node))
    }

    fn need_promotion(&self) -> bool {
        self.values.len() >= crate::LEAF_CAP
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
        results: Arc<Mutex<ResultSet<ResultEntry>>>,
    ) -> Result<(), AnyError> {
        let res = self
            .values
            .iter()
            .map(|x| (distance.distance(x, &i), x.clone()))
            .filter(|(x, _)| *x <= radius)
            .map(|x| ResultEntry::from_tuple(x))
            .collect::<Vec<ResultEntry>>();
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
