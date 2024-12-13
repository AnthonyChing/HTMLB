f = open(r'../stage 1/submissions/gaussian_C0.1_gamma0.1_all_data_78col.csv', 'r')
g = open(r'../stage 1/submissions/lightGBM.csv', 'r')
h = open(r'../stage 1/submissions/RF-100000-1126.csv', 'r')

aaron = 0
anthony = 0
uranus = 0

f.readline()
g.readline()
h.readline()
j = open('blend.csv', 'w')
j.write('id,home_team_win\n')
for i in range(6185):
    linef = 1 if f.readline().split(',')[1] == 'True\n' else 0
    lineg = 1 if g.readline().split(',')[1] == 'True\n' else 0
    lineh = 1 if h.readline().split(',')[1] == 'True\n' else 0
    result = (linef + lineg + lineh)
    if(linef + lineg + lineh >= 2):
        j.write(f"{i},True\n")
    else:
        j.write(f"{i},False\n")

    if result >= 2:
        if linef == 0:
            aaron += 1
        if lineg == 0:
            anthony += 1
        if lineh == 0:
            uranus += 1
    else:
        if linef == 1:
            aaron += 1
        if lineg == 1:
            anthony += 1
        if lineh == 1:
            uranus += 1

print("Aaron", aaron, "Anthony", anthony, "Uranus", uranus)
print((aaron + anthony + uranus)/ 6185)
print(anthony/6185)
print(aaron/6185)
print(uranus/6185)