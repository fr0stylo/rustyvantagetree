use std::fmt::Debug;

pub trait BranchPruner<T>: Debug + Sync + Send {
    fn prune(&self, dist: T, radius: T, threshold: T) -> (bool, bool);
}

#[derive(Debug)]
pub(crate) struct ExactShaker;

impl BranchPruner<usize> for &ExactShaker {
    fn prune(&self, dist: usize, radius: usize, threshold: usize) -> (bool, bool) {
        (
            dist.saturating_sub(radius) <= threshold,
            dist.saturating_add(radius) > threshold,
        )
    }
}

#[derive(Debug)]
pub(crate) struct AproximateShaker;

impl BranchPruner<usize> for &AproximateShaker {
    fn prune(&self, dist: usize, _: usize, threshold: usize) -> (bool, bool) {
        (
            dist <= threshold.saturating_add(1),
            dist > threshold.saturating_sub(1),
        )
    }
}
