





def testen(eins, zwei, drei):
    return eins+zwei+drei


test1 = 0
test2 = 5
states = 3


ergebnis = testen(
    eins=test1,
    zwei=test2,
    drei=(states+2),
)

print(ergebnis)