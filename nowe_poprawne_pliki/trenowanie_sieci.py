import numpy as np
import os
import moje_klasy

plik_liczba_gier = open("liczba_gier.txt","r")
liczba_rozegranych_gier = plik_liczba_gier.readlines()
liczba_rozegranych_gier = int(liczba_rozegranych_gier[0])
plik_liczba_gier.close()

p_1 = moje_klasy.plansza()
g_L = moje_klasy.gracz_lewy()
g_P = moje_klasy.gracz_prawy('Prawy')
g_L.stworz_siec()
g_P.stworz_siec()

folder_gracz_lewy = 'model_gracz_lewy'
folder_gracz_prawy = 'model_gracz_prawy'

if os.path.exists('./' + folder_gracz_lewy) and os.path.exists('./' + folder_gracz_prawy):
    print('Wczytywanie modeli')
    g_L.wczytaj_model(folder_gracz_lewy)
    g_P.wczytaj_model(folder_gracz_prawy)

gracze = [g_P,g_L]
docelowa_liczba_gier = 40000

for i in range(liczba_rozegranych_gier, docelowa_liczba_gier + 1):
    eps = 1 - i/(docelowa_liczba_gier + 1)
    #eps = 0.1
    print('gra' + str(i))
    p_1 = moje_klasy.plansza()
    kolejnosc = np.random.randint(0,2) == 1
    koniec = False
    while koniec == False:
        gracz_temp = gracze[int(kolejnosc)]
        #p_1.wyswietl()
        #time.sleep(1)
        print('Ruch gracza: ' + gracz_temp.imie)  
        #time.sleep(1)
        #print(p_1.pozycja)
        #print(p_1.indeks_pilki)
        kolejny_ruch,nagroda = gracz_temp.ruch(p_1,eps,0.999)
        
        
        if nagroda != 0: 
            koniec = True
            #gracze[int(kolejnosc)].finalna_nagroda(nagroda)
            gracze[int(not kolejnosc)].finalna_nagroda(-1 * nagroda)
            #gracze[int(not kolejnosc)].finalna_nagroda(-1 * nagroda)
            g_L.resetuj()
            g_P.resetuj()
        
        if not kolejny_ruch:
            kolejnosc = not kolejnosc
    
    if i % 100 == 0:
        g_L.model.save(folder_gracz_lewy)
        g_P.model.save(folder_gracz_prawy)
        plik_liczba_gier = open("liczba_gier.txt","w")
        plik_liczba_gier.write(str(i+1))
        plik_liczba_gier.close()
