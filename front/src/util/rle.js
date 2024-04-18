// Decode from RLE to Binary Mask
// (pass false to flat argument if you need 2d matrix output)
export function decodeCocoRLE([rows, cols], counts, flat = true) {
    let pixelPosition = 0,
        binaryMask

    if (flat) {
        binaryMask = Array(rows * cols).fill(0)
    } else {
        binaryMask = Array.from({length: rows}, (_) => Array(cols).fill(0))
    }

    for (let i = 0, rleLength = counts.length; i < rleLength; i += 2) {
        let zeros = counts[i],
            ones = counts[i + 1] ?? 0

        pixelPosition += zeros

        while (ones > 0) {
            const rowIndex = pixelPosition % rows,
                colIndex = (pixelPosition - rowIndex) / rows

            if (flat) {
                const arrayIndex = rowIndex * cols + colIndex
                binaryMask[arrayIndex] = 1
            } else {
                binaryMask[rowIndex][colIndex] = 1
            }

            pixelPosition++
            ones--
        }
    }

    if (!flat) {
        console.log("Result matrix:")
        binaryMask.forEach((row, i) => console.log(row.join(" "), `- row ${i}`))
    }

    return binaryMask
}

export function decodeMask(size, counts) {
    const mask = new Uint8Array(size[0] * size[1]);

    let p = 0;
    let c = 0;
    let zeros = 0;
    let ones = 0;

    for (let i = 0; i < counts.length; i++) {
        let count = counts[i];

        if (zeros + ones + count > mask.length) {
            break;
        }

        if (i % 2 === 0) {
            zeros += count;
        } else {
            ones += count;

            while (count > 0) {
                mask[p++] = 1;
                count--;
            }
        }
    }

    return mask;
}

/**
 * 将
 * @param rows
 * @param cols
 * @param counts
 * @returns {Uint8Array}
 */
export function decodeRleCounts([rows, cols], counts) {
    let arr = new Uint8Array(rows * cols)
    let i = 0
    let flag = 0
    for (let k of counts) {
        while (k-- > 0) {
            arr[i++] = flag
        }
        flag = (flag + 1) % 2
    }
    return arr
}