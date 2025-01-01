use std::fmt::Debug;

// pub trait CostFunction: Debug + Clone {
//   /// apply g function to z (g(z))
//   fn cost(actual: &Vec<f64>, predicted: &Vec<f64>) -> f64;

//   /// apply g' function to z relativee to z (g'(z) or dg(z)/dz)
//   fn derivative(value: &Vec<f64>, actual: &Vec<f64>) -> Vec<f64>;
// }

// #[derive(Debug, Clone)]
// pub struct MSE;

// impl CostFunction for MSE {
//   fn cost(actual: &Vec<f64>, predicted: &Vec<f64>) -> f64 {
//     assert_eq!(actual.len(), predicted.len());
//     let size = actual.len();
//     let mut result = 0.0;
//     for i in 0..size {
//       // println!("{} {}",actual[i],predicted[i]);
//       result += (actual[i] - predicted[i]).powf(2.0);
//     }
//     result /= size as f64;
//     return result;
//   }
//   fn derivative(activations: &Vec<f64>, actual: &Vec<f64>) -> Vec<f64> {}
// }

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
