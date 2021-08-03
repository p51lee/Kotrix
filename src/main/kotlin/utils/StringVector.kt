package utils

import complex.ComplexTensor
import real.Tensor

internal class StringVector(val stringData: ArrayList<String>) {
    override fun toString(): String {
        var retStr = ""
        stringData.forEachIndexed {index, value ->
            retStr += value
            if (index != stringData.lastIndex) retStr += "\n"
        }
        return retStr
    }

    fun concatHorizontal(other: StringVector): StringVector {
        return if (stringData.size != other.stringData.size) throw IllegalArgumentException("StringVector: invalid Size")
        else {
            val newStringData = arrayListOf<String>()
            stringData.forEachIndexed {index, str -> newStringData.add(str + "  " + other.stringData[index])}
            StringVector(newStringData)
        }
    }

    fun concatVertical(other: StringVector): StringVector {
        return StringVector((stringData + arrayListOf(" ".repeat(other.stringData[0].length)) + other.stringData) as ArrayList<String>)
    }

    fun rawConcatVertical(other: StringVector): StringVector {
        return StringVector((stringData + other.stringData) as ArrayList<String>)
    }

    companion object {

        fun build(tensor: Tensor): StringVector {
            val shape = tensor.shape
            val dim = tensor.dim
            return when (dim) {
                1 -> {
                    val stringData = arrayListOf<String>()
                    var data = "[ "
                    for (i in 0 until shape[0]) {
                        val value = tensor[intArrayOf(i)]
                        data += value.toFormattedString()
                    }
                    data += " ]"
                    stringData.add(data)
                    StringVector(stringData)
                }
                2 -> {
                    val stringData = arrayListOf<String>()
                    for (i in 0 until shape[0]) {
                        var data = "[ "
                        for (j in 0 until shape[1]) {
                            val value = tensor[intArrayOf(i, j)]
                            data += value.toFormattedString()
                        }
                        data += " ]"
                        stringData.add(data)
                    }
                    StringVector(stringData)
                }
                else -> {
                    val leftBracketData = arrayListOf<String>()
                    val rightBracketData = arrayListOf<String>()
                    val upperBlankData =  arrayListOf<String>()
                    val lowerBlankData =  arrayListOf<String>()
                    when {
                        dim % 2 == 0 -> {
                            val height = shape.foldRightIndexed(1) { index, i, acc ->
                                when {
                                    index % 2 == 1                  -> acc
                                    index == shape.lastIndex - 1    -> acc * i
                                    else                            -> acc * i + i + 1
                                }
                            }
                            repeat(height) { leftBracketData.add("["); rightBracketData.add("]") }
                            val leftBracket = StringVector(leftBracketData)
                            val rightBracket = StringVector(rightBracketData)
                            val bodyStringVector = (1 until shape[0]).fold(
                                (1 until shape[1]).fold( build(tensor[0L][0L]) ) { acc2, j ->
                                    acc2.concatHorizontal(build(tensor[0L][j.toLong()]))
                                }
                            ) { acc1, i ->
                                acc1.concatVertical(
                                    (1 until shape[1]).fold( build(tensor[i.toLong()][0L]) ) { acc2, j ->
                                        acc2.concatHorizontal(build(tensor[i.toLong()][j.toLong()]))
                                    }
                                )
                            }
                            val bodyUpperWidth = bodyStringVector.stringData[0].length
                            val bodyLowerWidth = bodyStringVector.stringData.last().length
                            upperBlankData.add(" ".repeat(bodyUpperWidth))
                            lowerBlankData.add(" ".repeat(bodyLowerWidth))
                            val upperBlank = StringVector(upperBlankData)
                            val lowerBlank = StringVector(lowerBlankData)

                            val bodyStringVectorWithPadding = upperBlank.rawConcatVertical(bodyStringVector).rawConcatVertical(lowerBlank)
                            leftBracket.concatHorizontal(bodyStringVectorWithPadding).concatHorizontal(rightBracket)
                        }
                        else -> {
                            val height = shape.foldRightIndexed(1) { index, i, acc ->
                                when {
                                    index == 0 -> acc + 2
                                    index % 2 == 0 -> acc
                                    index == shape.lastIndex - 1 -> acc * i
                                    else -> acc*i + i + 1
                                }
                            }
                            repeat(height) { leftBracketData.add("["); rightBracketData.add("]") }
                            val leftBracket = StringVector(leftBracketData)
                            val rightBracket = StringVector(rightBracketData)
                            val bodyStringVector = (1 until shape[0]).fold(build(tensor[0L])) { acc, i ->
                                acc.concatHorizontal(build(tensor[i.toLong()]))
                            }
                            val bodyUpperWidth = bodyStringVector.stringData[0].length
                            val bodyLowerWidth = bodyStringVector.stringData.last().length
                            upperBlankData.add(" ".repeat(bodyUpperWidth))
                            lowerBlankData.add(" ".repeat(bodyLowerWidth))
                            val upperBlank = StringVector(upperBlankData)
                            val lowerBlank = StringVector(lowerBlankData)

                            val bodyStringVectorWithPadding = upperBlank.rawConcatVertical(bodyStringVector).rawConcatVertical(lowerBlank)
                            leftBracket.concatHorizontal(bodyStringVectorWithPadding).concatHorizontal(rightBracket)
                        }
                    }
                }
            }
        }

        fun build(complexTensor: ComplexTensor): StringVector {
            val dim = complexTensor.dim
            val shape = complexTensor.shape
            return when (dim) {
                1 -> {
                    val stringData = arrayListOf<String>()
                    var data = "[ "
                    for (i in 0 until shape[0]) {
                        val value = complexTensor[intArrayOf(i)]
                        data += value.toString()
                    }
                    data += " ]"
                    stringData.add(data)
                    StringVector(stringData)
                }
                2 -> {
                    val stringData = arrayListOf<String>()
                    for (i in 0 until shape[0]) {
                        var data = "[ "
                        for (j in 0 until shape[1]) {
                            val value = complexTensor[intArrayOf(i, j)]
                            data += value.toString()
                        }
                        data += " ]"
                        stringData.add(data)
                    }
                    StringVector(stringData)
                }
                else -> {
                    val leftBracketData = arrayListOf<String>()
                    val rightBracketData = arrayListOf<String>()
                    val upperBlankData =  arrayListOf<String>()
                    val lowerBlankData =  arrayListOf<String>()
                    when {
                        dim % 2 == 0 -> {
                            val height = shape.foldRightIndexed(1) { index, i, acc ->
                                when {
                                    index % 2 == 1                  -> acc
                                    index == shape.lastIndex - 1    -> acc * i
                                    else                            -> acc * i + i + 1
                                }
                            }
                            repeat(height) { leftBracketData.add("["); rightBracketData.add("]") }
                            val leftBracket = StringVector(leftBracketData)
                            val rightBracket = StringVector(rightBracketData)
                            val bodyStringVector = (1 until shape[0]).fold(
                                (1 until shape[1]).fold(build(complexTensor[0L][0L])) { acc2, j ->
                                    acc2.concatHorizontal(build(complexTensor[0L][j.toLong()]))
                                }
                            ) { acc1, i ->
                                acc1.concatVertical(
                                    (1 until shape[1]).fold(build(complexTensor[i.toLong()][0L])) { acc2, j ->
                                        acc2.concatHorizontal(build(complexTensor[i.toLong()][j.toLong()]))
                                    }
                                )
                            }
                            val bodyUpperWidth = bodyStringVector.stringData[0].length
                            val bodyLowerWidth = bodyStringVector.stringData.last().length
                            upperBlankData.add(" ".repeat(bodyUpperWidth))
                            lowerBlankData.add(" ".repeat(bodyLowerWidth))
                            val upperBlank = StringVector(upperBlankData)
                            val lowerBlank = StringVector(lowerBlankData)

                            val bodyStringVectorWithPadding = upperBlank.rawConcatVertical(bodyStringVector).rawConcatVertical(lowerBlank)
                            leftBracket.concatHorizontal(bodyStringVectorWithPadding).concatHorizontal(rightBracket)
                        }
                        else -> {
                            val height = shape.foldRightIndexed(1) { index, i, acc ->
                                when {
                                    index == 0 -> acc + 2
                                    index % 2 == 0 -> acc
                                    index == shape.lastIndex - 1 -> acc * i
                                    else -> acc*i + i + 1
                                }
                            }
                            repeat(height) { leftBracketData.add("["); rightBracketData.add("]") }
                            val leftBracket = StringVector(leftBracketData)
                            val rightBracket = StringVector(rightBracketData)
                            val bodyStringVector = (1 until shape[0]).fold(build(complexTensor[0L])) { acc, i ->
                                acc.concatHorizontal(build(complexTensor[i.toLong()]))
                            }
                            val bodyUpperWidth = bodyStringVector.stringData[0].length
                            val bodyLowerWidth = bodyStringVector.stringData.last().length
                            upperBlankData.add(" ".repeat(bodyUpperWidth))
                            lowerBlankData.add(" ".repeat(bodyLowerWidth))
                            val upperBlank = StringVector(upperBlankData)
                            val lowerBlank = StringVector(lowerBlankData)

                            val bodyStringVectorWithPadding = upperBlank.rawConcatVertical(bodyStringVector).rawConcatVertical(lowerBlank)
                            leftBracket.concatHorizontal(bodyStringVectorWithPadding).concatHorizontal(rightBracket)
                        }
                    }
                }
            }
        }
    }
}