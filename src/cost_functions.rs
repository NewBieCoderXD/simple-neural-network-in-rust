use std::fmt::Debug;

pub trait CostFunction: Debug{
  /// apply g function to z (g(z))
  fn activate(actual:Vec<f64>,predicted: Vec<f64>) -> f64;

  /// apply g' function to z relativee to z (g'(z) or dg(z)/dz)
  fn derivative(value: &Vec<f64>) -> Vec<f64>;
}

#[derive(Debug,Clone)]
pub struct MSE;

impl CostFunction for MSE{
  fn activate(actual:Vec<f64>,predicted: Vec<f64>) -> f64{
    assert_eq!(actual.len(),predicted.len());
    let size = actual.len();
    let mut result = 0.0;
    for i in 0..size{
      result+=(actual[i]-predicted[i]).powf(2.0);
    }
    result/=size as f64;
    return result;
  }
  fn derivative(value: &Vec<f64>) -> Vec<f64> {
      todo!();
  }
}