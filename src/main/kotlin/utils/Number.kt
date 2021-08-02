package utils

import complex.*
import real.ColumnVector
import real.Matrix
import real.RowVector
import real.Tensor

val Number.I: ComplexDouble
    get() = ComplexDouble(0, toDouble())

val Number.R: ComplexDouble
    get() = ComplexDouble(toDouble(), 0)


fun Number.toFormattedString(): String {
    val thisToDouble = this.toDouble()
    return when {
        thisToDouble >= 1000 -> " %.0f ".format(thisToDouble)
        thisToDouble >= 100 -> " %.0f. ".format(thisToDouble)
        thisToDouble >= 10 -> " %.1f ".format(thisToDouble)
        thisToDouble == -0.0 -> " 0.00 "
        thisToDouble >= 0 -> " %.2f ".format(thisToDouble)
        thisToDouble > -10 -> "%.2f ".format(thisToDouble)
        thisToDouble > -100 -> "%.1f ".format(thisToDouble)
        thisToDouble > -1000 -> "%.0f. ".format(thisToDouble)
        else -> "%.0f ".format(thisToDouble)
    }
}

operator fun Number.times(other: Tensor): Tensor {
    val newData = DoubleArray(other.size) { other.data[it] * this.toDouble() }
    return Tensor(other.shape, newData)
}

operator fun Number.times(other: ComplexTensor): ComplexTensor {
    val newData = Array(other.size) { other.data[it] * this.toDouble() }
    return ComplexTensor(other.shape, newData)
}

operator fun Number.times(other: ComplexMatrix): ComplexMatrix {
    val newData = Array(other.size) { other.data[it] * this.toDouble() }
    return ComplexMatrix(other.rows, other.cols, newData)
}

operator fun Number.times(other: ComplexRowVector): ComplexRowVector {
    val newData = Array(other.size) { other.data[it] * this.toDouble() }
    return ComplexRowVector(other.length, newData)
}

operator fun Number.times(other: ComplexColumnVector): ComplexColumnVector {
    val newData = Array(other.size) { other.data[it] * this.toDouble() }
    return ComplexColumnVector(other.length, newData)
}

operator fun Number.times(other: Matrix): Matrix {
    val newData = DoubleArray(other.rows * other.cols) {
        val rowIndex = it / other.cols
        val colIndex = it % other.cols
        this.toDouble() * other[rowIndex, colIndex]
    }
    return Matrix(other.rows, other.cols, newData)
}

operator fun Number.times(other: RowVector): RowVector {
    val newData = DoubleArray(other.length) {
        this.toDouble() * other[it]
    }
    return RowVector(other.length, newData)
}

operator fun Number.times(other: ColumnVector): ColumnVector {
    val newData = DoubleArray(other.length) {
        this.toDouble() * other[it]
    }
    return ColumnVector(other.length, newData)
}

operator fun Number.plus(other: ComplexDouble) = other + this

operator fun Number.minus(other: ComplexDouble) = -other + this

operator fun Number.times(other: ComplexDouble) = other * this

operator fun Number.div(other: ComplexDouble) = this.R / other

fun Number.toComplexDouble() = this.R

fun Array<ComplexDouble>.sum(): ComplexDouble {
    var sum = 0.R
    for (element in this) {
        sum += element
    }
    return sum
}