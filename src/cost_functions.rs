use std::fmt::Debug;

#[derive(Debug, Clone)]
pub enum CostFunction {
  MeanSquaredError,
}

impl CostFunction {
  pub fn derivative(&self, activations: &Vec<f64>, actual: &Vec<f64>) -> Vec<f64> {
    match self {
      CostFunction::MeanSquaredError => {
        assert_eq!(activations.len(), actual.len());
        activations
          .iter()
          .zip(actual.iter())
          .map(|(activation, actual)| 2.0 * (activation - actual) / activations.len() as f64)
          .collect()
      }
    }
  }
  pub fn cost(&self, actual: &Vec<f64>, predicted: &Vec<f64>) -> f64 {
    match self {
      CostFunction::MeanSquaredError => {
        assert_eq!(actual.len(), predicted.len());
        let size = actual.len();
        let mut result = 0.0;
        for i in 0..size {
          // println!("{} {}",actual[i],predicted[i]);
          result += (actual[i] - predicted[i]).powf(2.0);
        }
        result /= size as f64;
        return result;
      }
    }
  }
}
