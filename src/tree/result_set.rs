use std::collections::BinaryHeap;

#[derive(Eq, PartialEq, Debug, Clone)]
pub struct ResultEntry {
    value: Vec<u8>,
    distance: usize,
}

impl ResultEntry {
    pub fn from_tuple(i: (usize, Vec<u8>)) -> Self {
        ResultEntry {
            value: i.1,
            distance: i.0,
        }
    }
}

impl PartialOrd for ResultEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ResultEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance.cmp(&other.distance)
    }
}

pub struct ResultSet<V>
where
    V: Ord,
{
    results: BinaryHeap<V>,
    max_size: usize,
}

impl<V> ResultSet<V>
where
    V: Ord,
{
    pub fn new(max_size: usize) -> Self {
        ResultSet {
            results: BinaryHeap::new(),
            max_size: max_size,
        }
    }

    pub fn push(&mut self, ord: V) {
        self.results.push(ord);
        if self.max_size < self.results.len() {
            self.results.pop();
        }
    }

    pub fn extend(&mut self, ords: Vec<V>) {
        for ord in ords {
            self.results.push(ord);
            if self.max_size < self.results.len() {
                self.results.pop();
            }
        }
    }

    pub fn collect(&self) -> Vec<V>
    where
        V: Clone,
    {
        self.results.iter().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;

    #[bench]
    fn bench_tree_result_set(b: &mut Bencher) {
        let mut results = ResultSet::new(10);

        b.iter(|| results.push(ResultEntry::from_tuple((1, vec![123u8, 123u8]))));
    }
}
