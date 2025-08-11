#!/usr/bin/env python3

# python3 origami_test.py -m 2048 -n 2048 -k 2048 --transA T --transB N --element_size 1 --debug --print

# miDataType numbers:
# Float: 0
# Double: 1
# ComplexFloat: 2
# ComplexDouble: 3
# Half: 4
# Int8x4: 5
# Int32: 6
# BFloat16: 7
# Int8: 8
# Int64: 9
# XFloat32: 10
# Float8_fnuz: 11
# BFloat8_fnuz: 12
# Float8BFloat8_fnuz: 13
# BFloat8Float8_fnuz: 14
# Float8: 15
# BFloat8: 16
# Float8BFloat8: 17
# BFloat8Float8: 18
# Float6: 19
# Float4: 20

datatypes = {
0 : 'S_',
1 : 'D_', 
4 : 'H_',
7 : 'B_',
10 : 'X_',
15 : 'F8_',
}

MatInst = {
"F8_gfx950": [
    (4,4,4,16), #gfx942
    (16,16,128,1), #gfx950
    (32,32,64,1) #gfx950
    ],     
"H_gfx950":[
    # (4,4,4,16), #gfx942
    #[16,16,4,4] # never use 16x16x4x4
    #[16,16,16,1] #gfx942
    #[32,32,4,2] # never use 32x32x4x2
    #[32,32,8,1] #gfx942                          
    (16,16,32,1), #gfx950
    (32,32,16,1) #gfx950
    ],
"B_gfx950":[
    (4,4,4,16), #gfx942
    #[16,16,4,4] # never use 16x16x4x4
    #[16,16,16,1] #gfx942
    #[32,32,4,2] # never use 32x32x4x2
    #[32,32,8,1] #gfx942                          
    (16,16,32,1), #gfx950
    # (32,32,16,1) #gfx950
    ],
"S_gfx950":[
    (16,16,4,1),
    (32,32,2,1)
    ],
"X_gfx950": [
    (32,32,4,1),
    (16,16,8,1)
    ],
"D_gfx950":[
    (16,16,4,1)
    ],
# "C_gfx950": [
#   (16,16,4,1)
#     ],  
# "Z_gfx950":[
#   (16,16,4,1)
#     ],
# "I8_gfx950": [
#   (32,32,16,1),
#   (16,16,32,1),
#   (4,4,4,16)
# ],
}
LIST_OF_WAVEs_TO_INCLUDE = [[4, 1], [2, 2], [1, 4], [1, 2], [2, 1], [1, 1]]
MIN_MT0 = MIN_MT1 = 16
MAX_MT0 = MAX_MT1 = 512

import argparse
import origami
import csv
import os


def parseArguments():
    parser = argparse.ArgumentParser(description="""Test Origami.""")
    parser.add_argument("-m", type=int, default=8192)
    parser.add_argument("-n", type=int, default=8192)
    parser.add_argument("-b", type=int, default=1)
    parser.add_argument("-k", type=int, default=8192)
    parser.add_argument("--transA", type=bool, default=True)
    parser.add_argument("--transB", type=bool, default=False)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--miDataType", type=int, default=4) # see the comment for valid numbers
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--print", action="store_true")
    parser.add_argument("--wgm", type=int, default=6)
    parser.add_argument("--sizes", type=bool, default=False) # to load the sizes from a csv file. m/n/k will be ignored if True
    parser.add_argument("--path", type=str, default="./sizes.csv")  # path to the csv file. Fails if sizes is true, and path or file does not exist.
    parser.add_argument("--arch", type=str, default="gfx950")  # arch

    return parser.parse_args()

def createTileList(gemmType):
    # list of MIs for each datatype:
    bm_max = 0
    tile_list = set()
    for MI in MatInst[gemmType]:
        for bm in range(bm_max + 1):
            MIBlockM = 2 ** bm

            for wave in LIST_OF_WAVEs_TO_INCLUDE:
                waveTileM = 0
                waveTileN = 0

                while True:
                    waveTileM+=1
                    waveTileN=0
                    MatrixInstM = MI[0] * MIBlockM
                    MT0 = MatrixInstM * waveTileM * wave[0]
                    if MT0 < MIN_MT0:
                        continue
                    if MT0 > MAX_MT0:
                        break

                    while True:
                        waveTileN+=1
                        MatrixInstN = MI[1] / MIBlockM * MI[3]
                        MT1 = int(MatrixInstN * waveTileN * wave[1])

                        if MT1 < MIN_MT1:
                            continue
                        if MT1 > MAX_MT1:
                            break

                        # LDS size check for LSU
                        LSU = max(1, 4//wave[0]//wave[1])
                        if LSU > 1 and MT0*MT1*4*LSU > 256*256:
                            continue

                        if MT0*MT1 > 256*256:
                            continue
                        for DU in [32, 64, 128, 256, 512]:
                            tile_list.add((MT0, MT1, DU, MI[0], MI[1], MI[2], 1))

    return [tile for tile in tile_list]

def main():
    args = parseArguments()

    hardware = origami.getHardwareForDevice(args.device)

    if (args.miDataType not in datatypes):
        raise(" Wrong or not supported miDataType.")
    
    gemmType = datatypes[args.miDataType] + args.arch
    if (gemmType not in MatInst):    
        raise("Use a valid GEMM: B_gfx950, F_gfx950, F8_gfx950, S_gfx950, X_gfx950, D_gfx950")
    element_size = 0
    if (args.miDataType == 15):
        element_size = 1
    elif (args.miDataType == 7 or args.miDataType == 4):
        element_size = 2
    elif (args.miDataType == 0 or args.miDataType == 10):
        element_size = 4
    elif (args.miDataType == 1):
        element_size = 8

    tile_list = createTileList(gemmType)

    tile_list =[(256, 256, 32, 16, 16, 32, 1)]

    print(" Number of unique tiles: ", len(tile_list))

    if (args.sizes and not os.path.exists(args.path)):
        raise(" The size file does not exist.")
    if (args.sizes):
        with open("macrotile_fromOrigami.txt",'w') as file: # for the record
            for tile in tile_list:
                file.write(f'{tile}\n')

    if args.print:
        hardware.print()

    if (args.sizes): # sizes from a file
      with open(args.path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            M = int(row[0])
            N = int(row[1])
            B = int(row[2])
            K = int(row[3])

            print(" size: ", M, N, K)
            ret = origami.select_best_macro_tile_size(
                M,
                N,
                K,
                B,
                args.transA,
                args.transB,
                hardware,
                tile_list,
                element_size * 8,
                element_size * 8,
                element_size * 8,
                # args.miDataType,
                origami.DataType.BFloat16,
                0,
                0.8,
                args.debug,
                args.print,
                args.wgm,
            )
            print(f"{M},{N},{B},{K},{ret[0]}")
    else: # unique size from terminal
        M = args.m
        N = args.n
        K = args.k
        B = args.b
        print(" size: ", args.m, args.n, args.k)
        ret = origami.select_best_macro_tile_size(
            args.m,
            args.n,
            args.k,
            args.b,
            args.transA,
            args.transB,
            hardware,
            tile_list,
            element_size * 8,
            element_size * 8,
            element_size * 8,
            # args.miDataType,
            origami.DataType.BFloat16,
            0,
            0.8,
            args.debug,
            args.print,
            args.wgm,
        )
        print(f"The best combo for [{M}, {N}, {B}, {K}] is: {ret[0]}")
        print(" full list: \n", ret)

    if args.print:
        hardware.print_debug_info()

    return 0


if __name__ == "__main__":
    exit(main())
