from numpy import random

"""purchase probability (E) and age (F) depends on each other"""
random.seed(0)

totals = {20: 0, 30: 0, 40: 0, 50: 0, 60: 0, 70: 0}
purchases = {20: 0, 30: 0, 40: 0, 50: 0, 60: 0, 70: 0}
total_purchases = 0

for _ in range(100000):
    age_decade = random.choice([20, 30, 40, 50, 60, 70])
    purchase_probability = float(age_decade) / 100.0
    totals[age_decade] += 1
    if random.random() < purchase_probability:
        total_purchases += 1
        purchases[age_decade] += 1

print('Totals:', totals)
print('Purchases:', purchases)
print('Total Purchase:', total_purchases)

# Compute P(E|F), E = Purchase, F = age in 30's
PEF = float(purchases[30]) / float(totals[30])
print("P(purchases | 30s) : ", PEF)

# P(F) -> Probability of being 30 in the dataset
PF = float(totals[30]) / 100000.0
print("P(30's) : ", PF)

# P(E) -> overall probability of buying something
PE = float(total_purchases) / 100000.0
print("P(Purchases) : ", PE)

# If E and F were independent, we would expect P(E|F) to be about the same as P(E). P(E)=0.45, P(E|F)=0.299
# So, E and F are dependent

# P(E) * P(F)
print("P(30's)P(Purchases) : ", PE * PF)

# P(E ∩ F) -> Probability of being 30 and buying something
print("P(30's ∩ Purchases) : ", float(purchases[30]) / 100000.0)

# P(E|F) = P(E ∩ F) / P(F)
print("P(purchases | 30s) = P(30's ∩ Purchases) / P(30's) : ", float(purchases[30]) / 100000.0 / PF)
print()

"""purchase probability (E) and age (F) does not depend on each other (independent)"""
random.seed(0)

totals = {20: 0, 30: 0, 40: 0, 50: 0, 60: 0, 70: 0}
purchases = {20: 0, 30: 0, 40: 0, 50: 0, 60: 0, 70: 0}
total_purchases = 0

for _ in range(100000):
    age_decade = random.choice([20, 30, 40, 50, 60, 70])
    purchase_probability = 0.4
    totals[age_decade] += 1
    if random.random() < purchase_probability:
        total_purchases += 1
        purchases[age_decade] += 1

print('Totals:', totals)
print('Purchases:', purchases)
print('Total Purchase:', total_purchases)

# Compute P(E|F), E = Purchase, F = age in 30's
PEF = float(purchases[30]) / float(totals[30])
print("P(purchases | 30s) : ", PEF)

# P(F) -> Probability of being 30 in the dataset
PF = float(totals[30]) / 100000.0
print("P(30's) : ", PF)

# P(E) -> overall probability of buying something
PE = float(total_purchases) / 100000.0
print("P(Purchases) : ", PE)

# E and F are independent as P(E|F) to be about the same as P(E). P(E)=0.4003, P(E|F)=0.39876

# P(E) * P(F)
print("P(30's)P(Purchases) : ", PE * PF)

# P(E ∩ F) -> Probability of being 30 and buying something
print("P(30's ∩ Purchases) : ", float(purchases[30]) / 100000.0)

# P(E|F) = P(E ∩ F) / P(F)
print("P(purchases | 30s) = P(30's ∩ Purchases) / P(30's) : ", float(purchases[30]) / 100000.0 / PF)
