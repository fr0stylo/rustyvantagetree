use std::{fmt::Debug, simd::u8x8};

pub trait DistanceMetric<I: ?Sized, O>: Debug + Sync + Send {
    fn distance(&self, a: &I, b: &I) -> O;
}

#[derive(Debug)]
pub struct HammingDistance {
    chunk_size: usize,
}

impl HammingDistance {
    pub fn new(chunk_size: usize) -> Self {
        HammingDistance {
            chunk_size: chunk_size,
        }
    }
}

impl DistanceMetric<[u8], usize> for &HammingDistance {
    fn distance(&self, x: &[u8], y: &[u8]) -> usize {
        let len = x.len();
        let mut count = 0;

        let chunks = len / CHUNK_SIZE;
        for i in 0..chunks {
            let start = i * CHUNK_SIZE;

            let x_vec = u8x8::from_slice(&x[start..start + self.chunk_size]);
            let y_vec = u8x8::from_slice(&y[start..start + self.chunk_size]);

            let xor_result = x_vec ^ y_vec;

            for j in 0..self.chunk_size {
                count += xor_result[j].count_ones() as usize;
            }
        }

        // Handle remaining bytes
        let remainder_start = chunks * self.chunk_size;
        for i in remainder_start..len {
            count += (x[i] ^ y[i]).count_ones() as usize;
        }

        count
    }
}
impl DistanceMetric<[u8], usize> for HammingDistance {
    fn distance(&self, x: &[u8], y: &[u8]) -> usize {
        let len = x.len();
        let mut count = 0;

        let chunks = len / self.chunk_size;
        for i in 0..chunks {
            let start = i * self.chunk_size;

            let x_vec = u8x8::from_slice(&x[start..start + self.chunk_size]);
            let y_vec = u8x8::from_slice(&y[start..start + self.chunk_size]);

            let xor_result = x_vec ^ y_vec;

            for j in 0..self.chunk_size {
                count += xor_result[j].count_ones() as usize;
            }
        }

        // Handle remaining bytes
        let remainder_start = chunks * self.chunk_size;
        for i in remainder_start..len {
            count += (x[i] ^ y[i]).count_ones() as usize;
        }

        count
    }
}

const CHUNK_SIZE: usize = 8;

fn _hamming_distance(x: &[u8], y: &[u8]) -> usize {
    let len = x.len();
    let mut count = 0;

    let chunks = len / CHUNK_SIZE;
    for i in 0..chunks {
        let start = i * CHUNK_SIZE;

        let x_vec = u8x8::from_slice(&x[start..start + CHUNK_SIZE]);
        let y_vec = u8x8::from_slice(&y[start..start + CHUNK_SIZE]);

        let xor_result = x_vec ^ y_vec;

        for j in 0..CHUNK_SIZE {
            count += xor_result[j].count_ones() as usize;
        }
    }

    // Handle remaining bytes
    let remainder_start = chunks * CHUNK_SIZE;
    for i in remainder_start..len {
        count += (x[i] ^ y[i]).count_ones() as usize;
    }

    count
}

fn _hamming_distance_list(x: &[u8], y: &[u8]) -> usize {
    x.iter()
        .zip(y.iter())
        .fold(0, |acc, (x, y)| acc + (x ^ y).count_ones() as usize)
}

extern crate test;

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;

    #[bench]
    fn bench_hamming_distance_list(b: &mut Bencher) {
        let term1 = "34ec86d2".as_bytes().to_vec();
        let term2 = "35ef86d2".as_bytes().to_vec();

        b.iter(|| _hamming_distance_list(&term1, &term2));
    }

    #[bench]
    fn bench_hamming_distance_simd(b: &mut Bencher) {
        let term1 = "34ec86d2".as_bytes().to_vec();
        let term2 = "35ef86d2".as_bytes().to_vec();

        b.iter(|| _hamming_distance(&term1, &term2));
    }

    #[bench]
    fn bench_distance_metric_hamming(b: &mut Bencher) {
        let hamming = &HammingDistance::new(8);
        let x = "34ec86d2".as_bytes();
        let y = "34ec86dd".as_bytes();

        b.iter(|| hamming.distance(x, y));
    }
}
