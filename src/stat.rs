pub fn mean_and_standard_deviation(data: &Vec<f64>) -> (f64, f64) {
  let mut mean = 0.0;
  let mut m2 = 0.0;
  for i in 0..data.len() {
    let count = (i + 1) as f64;
    let delta = data[i] - mean;
    mean += delta / count;
    let delta2 = data[i] - mean;
    m2 += delta * delta2;
  }
  let n = data.len();
  return (mean, (m2 / (n - 1) as f64).sqrt());
}
