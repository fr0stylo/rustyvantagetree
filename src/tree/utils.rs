pub fn median_sorted_vec<T>(vec: &Vec<T>) -> T
where
    T: Clone,
{
    let len = vec.len();
    if len == 0 {
        panic!("Empty vector has no median");
    }

    if len % 2 == 1 {
        vec[len / 2].clone()
    } else {
        vec[len / 2 - 1].clone()
    }
}

pub fn to_bit_string(data: &[u8]) -> String {
    data.iter()
        .map(|byte| format!("{:08b}", byte))
        .collect::<Vec<String>>()
        .join("")
}
