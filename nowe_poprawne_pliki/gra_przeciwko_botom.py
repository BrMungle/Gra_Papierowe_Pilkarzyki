import numpy as np
import os
import moje_klasy

p_1 = moje_klasy.plansza()
g_L = moje_klasy.gracz_lewy()
g_P = moje_klasy.gracz_prawy('Prawy')
g_C = moje_klasy.gracz_czlowiek('Czlowiek')
#g_L.stworz_siec()
#g_P.stworz_siec()
folder_gracz_lewy = 'model_gracz_lewy'
folder_gracz_prawy = 'model_gracz_prawy'

if os.path.exists('./' + folder_gracz_lewy) and os.path.exists('./' + folder_gracz_prawy):
    print('Wczytywanie modeli')
    g_L.wczytaj_model(folder_gracz_lewy)
    g_P.wczytaj_model(folder_gracz_prawy)

nastepna = True

while(nastepna):
    gracze = [g_C]
    eps = 0
    #print('gra' + str(i))
    p_1 = moje_klasy.plansza()
    strona = ('L' == input('wpisz L aby grac po lewej stronie'))
    if strona :
        gracze = gracze + [g_P]
    else:
        gracze = gracze + [g_L]
        
    kolejnosc = np.random.randint(0,2) == 1
    koniec = False
    while koniec == False:
        gracz_temp = gracze[int(kolejnosc)]
        p_1.wyswietl()
        p_1.mozliwe_ruchy_klawiatura()
        #time.sleep(1)
        print('Ruch gracza: ' + gracz_temp.imie)  
        #time.sleep(1)
        #print(p_1.pozycja)
        #print(p_1.indeks_pilki)
        kolejny_ruch,nagroda = gracz_temp.wykonaj_najlepszy_ruch(p_1)
        
        
        if nagroda != 0: 
            koniec = True
            #gracze[int(kolejnosc)].finalna_nagroda(nagroda)
            #gracze[int(not kolejnosc)].finalna_nagroda(-1 * nagroda)
            g_L.resetuj()
            g_C.resetuj()
        
        if not kolejny_ruch:
            kolejnosc = not kolejnosc
        
    nastepna = ('1' == input('wpisz 1 aby zagrac jeszcze raz'))

