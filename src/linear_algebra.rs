use std::{default::Default, fmt::Display, ops::{Add, Mul}};

/// cross multiplying matrix, and vector.
/// matrix is on the left, while vector on the right
/// vector has one column
/// matrix(n x m) x vector(m x 1)
pub fn matrix_dot_vector<T>(matrix: &Vec<Vec<T>>,vector: &Vec<T>) -> Vec<T>
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

pub fn scalar_multiply_vector<T>(value:T, vector: &Vec<T>) -> Vec<T>
where T: Default + Mul<T, Output = T> + Copy {
  vector.iter().map(|&original| original*value).collect()
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
#[macro_export]
macro_rules! hadamard_product {
    ($($vectors:expr), +) => {
        hadamard_product(&[$(&$vectors,)*])
    };
}

pub fn hadamard_product<T>(vectors: &[&Vec<T>]) -> Vec<T>
where T: Default + Mul<T, Output = T> + Copy + Display {
  let mut itr = vectors.iter();
  let first = (*itr.next().unwrap()).clone();
  itr.fold(first,|acc: Vec<T>,vector: &&Vec<T>|{
    return acc
      .iter()
      .zip(vector.iter())
      .map(|(&lhs,&rhs)|{
        // println!("{} {}",lhs,rhs);
        return lhs*rhs
      })
      .collect()
  })
}

/// lhs(nx1) * rhsᵀ(1xm)
pub fn vector_dot_transposed_vector<T>(lhs_vector: &Vec<T>, rhs_vector: &Vec<T>) -> Vec<Vec<T>>
where T: Default + Mul<T, Output = T> + Copy + Display {
  let rows = lhs_vector.len();
  let columns = rhs_vector.len();

  (0..rows).map(|row|{
    (0..columns).map(|column|{
      rhs_vector[column]*lhs_vector[row]
    }).collect()
  }).collect()
}

pub fn add_matrices<T>(lhs_matrix: &Vec<Vec<T>>, rhs_matrix: &Vec<Vec<T>>) -> Vec<Vec<T>>
where T: Copy + Add<T,Output = T>{
  lhs_matrix.iter().zip(rhs_matrix.iter()).map(|(lhs_row,rhs_row)|{
    lhs_row.iter().zip(rhs_row.iter()).map(|(&lhs_column,&rhs_column)|{
      lhs_column+rhs_column
    }).collect()
  }).collect()
}

pub fn scalar_multiply_matrix<T>(scalar: T, matrix: &Vec<Vec<T>>) -> Vec<Vec<T>>
where T: Copy+Mul<T, Output = T>{
  matrix.iter().map(|row|{
    row.iter().map(|&element|{
      element*scalar
    }).collect()
  }).collect()
}