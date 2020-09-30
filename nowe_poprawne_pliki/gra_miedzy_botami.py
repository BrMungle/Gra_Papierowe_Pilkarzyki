import numpy as np
import os
import moje_klasy

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

nastepna = True

while(nastepna):
    eps = 0
    p_1 = moje_klasy.plansza()
    kolejnosc = np.random.randint(0,2) == 1
    koniec = False
    while koniec == False:
        gracz_temp = gracze[int(kolejnosc)]
        p_1.wyswietl()
        p_1.mozliwe_ruchy_klawiatura()
        print('Ruch gracza: ' + gracz_temp.imie)  
        kolejny_ruch,nagroda = gracz_temp.wykonaj_najlepszy_ruch(p_1)
        
        
        if nagroda != 0: 
            koniec = True
            g_L.resetuj()
            g_P.resetuj()
        
        if not kolejny_ruch:
            kolejnosc = not kolejnosc
        
    nastepna = ('1' == input('wpisz 1 aby zagrac jeszcze raz'))
