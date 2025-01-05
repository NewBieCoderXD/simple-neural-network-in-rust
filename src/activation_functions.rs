use std::fmt::Debug;

#[derive(Debug, Clone,Copy)]
pub enum ActivationFunction {
  Linear,
  Sigmoid,
  Tanh,
}

impl ActivationFunction {
  /// apply g function to z (g(z))
  pub fn activate(&self, value: &Vec<f64>) -> Vec<f64> {
    return match self {
      Self::Linear => value.iter().cloned().collect(),
      Self::Sigmoid => value
        .iter()
        .map(|&value| 1.0 / (1.0 + std::f64::consts::E.powf(-value)))
        .collect(),
      Self::Tanh => value
        .iter()
        .map(|&value| 2.0 / (1.0 + std::f64::consts::E.powf(-2.0 * value)) - 1.0)
        .collect(),
    };
  }

  /// apply g' function to z relativee to z (g'(z) or dg(z)/dz)
  pub fn derivative(&self, value: &Vec<f64>) -> Vec<f64> {
    return match self {
      Self::Linear => vec![1.0; value.len()],
      Self::Sigmoid => value
        .iter()
        .map(|&value| {
          let x = std::f64::consts::E.powf(-value);
          x / (1.0 + x).powi(2)
        })
        .collect(),
      Self::Tanh => value
        .iter()
        .map(|&value| {
          let x = std::f64::consts::E.powf(-2.0 * value);
          4.0 * x / (1.0 + x).powi(2)
        })
        .collect(),
    };
  }
}
