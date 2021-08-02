package utils

import complex.ComplexDouble
import complex.ComplexTensor
import real.Tensor
import kotlin.math.abs
import kotlin.math.pow

val Tensor.convCheck: Double
    get() = 10.0.pow(-5)

val Tensor.equalityValidation: Double
    get() = 10.0.pow(-4)

fun Tensor.pseudoEquals(x: Double, y: Double): Boolean {
    return abs(x - y) < equalityValidation
}

val ComplexTensor.convCheck: Double
    get() = 10.0.pow(-5)


val ComplexTensor.equalityValidation: Double
    get() = 10.0.pow(-4)

fun ComplexTensor.pseudoEquals(x: ComplexDouble, y: ComplexDouble): Boolean {
    return (x - y).abs() < equalityValidation
}