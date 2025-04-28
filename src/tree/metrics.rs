use std::{
    cmp::{max, min},
    ops::AddAssign,
    usize,
};

#[derive(Debug)]
pub struct TreeMetrics {
    nodes: usize,
    leaves: usize,
    elements: usize,
    max_depth: usize,
    min_depth: usize,
    steps: usize,
    avg_depth: f32,
}

impl TreeMetrics {
    pub fn new() -> Self {
        TreeMetrics {
            nodes: 0,
            leaves: 0,
            elements: 0,
            max_depth: 0,
            min_depth: usize::MAX,
            steps: 0,
            avg_depth: 0.0,
        }
    }

    pub fn leaf(size: usize) -> Self {
        TreeMetrics {
            nodes: 0,
            leaves: 1,
            elements: size,
            max_depth: 0,
            min_depth: 1,
            steps: 0,
            avg_depth: 0.0,
        }
    }
    pub fn node() -> Self {
        TreeMetrics {
            nodes: 1,
            leaves: 0,
            elements: 0,
            max_depth: 1,
            min_depth: usize::MAX,
            steps: 1,
            avg_depth: 0.0,
        }
    }

    pub fn update_depth(&mut self) {
        self.min_depth = self.min_depth.saturating_add(1);
        self.max_depth = self.max_depth.saturating_add(1);
    }
}

impl AddAssign for TreeMetrics {
    fn add_assign(&mut self, rhs: Self) {
        self.nodes += rhs.nodes;
        self.leaves += rhs.leaves;
        self.elements += rhs.elements;
        self.steps += rhs.steps + 1;

        self.max_depth = max(self.max_depth, rhs.max_depth);
        self.min_depth = min(self.min_depth, rhs.min_depth);
        self.avg_depth = self.steps as f32 / self.leaves as f32;
    }
}

pub trait TreeNodeMetrics {
    fn size(&self) -> TreeMetrics;
}
