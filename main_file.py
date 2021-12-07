#всему быть тут

import random

#тут должны быть все переменные и их значения
#штаммов всего 3, поэтому
M = 3
#число людей
N = 12000000
#множества переболевших штаммами
R_0 = [0]
R_1 = [0]
R_2 = [0]
R_3 = [0]
R_1_2 = [0]
R_1_3 = [0]
R_2_3 = [0]
R_1_2_3 = [0]
#множества переболевших одними и болеющих другим
R_0_J_1 = [0]
R_0_J_2 = [0]
R_0_J_3 = [0]
R_1_J_2 = [0]
R_1_J_3 = [0]
R_2_J_1 = [0]
R_2_J_3 = [0]
R_3_J_1 = [0]
R_3_J_2 = [0]
R_1_2_J_3 = [0]
R_1_3_J_2 = [0]
R_2_3_J_1 = [0]
#число погибших
D = [0]
#характеристики штаммов
#вероятность заболевания (последняя цифра - штамм, которым челвоек заболеет; до этого - штаммы, которые человек уже перенес)
betta_0_1 =  random.uniform(0.01, 0.10)
betta_0_2 =  random.uniform(0.01, 0.10)
betta_0_3 =  random.uniform(0.01, 0.10)
betta_1_2 = random.uniform(0.01, 0.10)
betta_1_3 = random.uniform(0.01, 0.10)
betta_2_1 = random.uniform(0.01, 0.10)
betta_2_3 = random.uniform(0.01, 0.10)
betta_3_1 = random.uniform(0.01, 0.10)
betta_3_2 = random.uniform(0.01, 0.10)
betta_1_2_3 = random.uniform(0.01, 0.10)
betta_1_3_2 =  random.uniform(0.01, 0.10)
betta_2_3_1 = random.uniform(0.01, 0.10)
#длительность болезни штаммом (последняя цифра - текущий штамм)
gamma_0_1 = random.uniform(0.03, 0.33)
gamma_0_2 = random.uniform(0.03, 0.33)
gamma_0_3 = random.uniform(0.03, 0.33)
gamma_1_2 =random.uniform(0.03, 0.33)
gamma_1_3 =random.uniform(0.03, 0.33)
gamma_2_1 =random.uniform(0.03, 0.33)
gamma_2_3 =random.uniform(0.03, 0.33)
gamma_3_1 =random.uniform(0.03, 0.33)
gamma_3_2 =random.uniform(0.03, 0.33)
gamma_1_2_3 =random.uniform(0.03, 0.33)
gamma_1_3_2 = random.uniform(0.03, 0.33)
gamma_2_3_1 =random.uniform(0.03, 0.33)
#вероятность выздоровления (последняя цифра - текущий штамм)
ksi_0_1 =  random.uniform(0.9, 0.99)
ksi_0_2 =  random.uniform(0.9, 0.99)
ksi_0_3 =  random.uniform(0.9, 0.99)
ksi_1_2 = random.uniform(0.9, 0.99)
ksi_1_3 = random.uniform(0.9, 0.99)
ksi_2_1 = random.uniform(0.9, 0.99)
ksi_2_3 = random.uniform(0.9, 0.99)
ksi_3_1 = random.uniform(0.9, 0.99)
ksi_3_2 = random.uniform(0.9, 0.99)
ksi_1_2_3 = random.uniform(0.9, 0.99)
ksi_1_3_2 =  random.uniform(0.9, 0.99)
ksi_2_3_1 = random.uniform(0.9, 0.99)

#ну вроде всё, дальше пишем код

