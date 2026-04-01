import os
from random import randint

outputFolder = "./output/"

def generateInputForLasVegas_primeField(n):
    F = GF(n)
    while True:
        a = F.random_element()
        b = F.random_element()
        discriminant = -16 * ((4 * (a * a * a)) + (27 * (b * b)))
        if discriminant == 0:
            continue
        E = EllipticCurve([a, b])
        G = E.abelian_group()
        gens = G.gens()
        if len(gens) == 1:
            ord = gens[0].order()
            if is_prime(ord):
                return F, E, G, a, b


def GeneratePrimeFieldInput(p):
    F, E, G, a, b = generateInputForLasVegas_primeField(p)

    ord_G = G.order()

    # Randomly generate m in range [1, ord_G - 1] to ensure m is never 0
    # and Q = m * P is never the point at infinity
    m = randint(1, int(ord_G) - 1)

    # Ensure both P and Q are not the point at infinity
    while True:
        P = E.random_element()
        if not P.is_zero():
            Q = m * P
            if not Q.is_zero():
                break

    # Ensure output directory exists before writing
    os.makedirs(outputFolder, exist_ok=True)

    fout = open(outputFolder + str(p) + "_.txt", 'w')

    print(str(p) + "\t" + str(m))
    fout.write(str(p) + "\t" + str(m) + "\n")

    print(str(a) + "\t" + str(b))
    fout.write(str(a) + "\t" + str(b) + "\n")

    print(str(P.xy()[0]) + "\t" + str(P.xy()[1]))
    fout.write(str(P.xy()[0]) + "\t" + str(P.xy()[1]) + "\n")

    print(str(Q.xy()[0]) + "\t" + str(Q.xy()[1]))
    fout.write(str(Q.xy()[0]) + "\t" + str(Q.xy()[1]) + "\n")

    print(str(ord_G))
    fout.write(str(ord_G) + "\n")

    fout.write("#")
    fout.close()
