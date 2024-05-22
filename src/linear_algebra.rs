use std::ops::{Add, Mul};

/// cross multiplying matrix, and vector.
/// matrix is on the left, while vector on the right
/// vector has one column
/// matrix(n x m) x vector(m x 1)
pub fn matrix_cross_vector<T>(matrix: &Vec<Vec<T>>,vector: &Vec<T>) -> Vec<T>
where T: Default + Add<T,Output = T> + Mul<T, Output = T> + Copy {
  // let matrix_rows = matrix.len();
  let matrix_columns = matrix[0].len();
  assert_eq!(matrix_columns,vector.len(),"invalid matrix and vector crossing size");
  matrix.iter().map(|row|{
    let init: T = Default::default();
    row.iter().zip(vector.iter()).fold(init,|sum,(&column_ele,&vec_ele)|{
      return sum+column_ele*vec_ele;
    })
  }).collect()
}

pub fn add_vector<T>(lhs_vector: &Vec<T>, rhs_vector: &Vec<T>) -> Vec<T>
where T: Default + Add<T, Output = T> + Copy {
  assert_eq!(lhs_vector.len(),rhs_vector.len());
  lhs_vector
  .iter()
  .zip(rhs_vector.iter())
  .map(|(&lhs,&rhs)| lhs+rhs)
  .collect()
}