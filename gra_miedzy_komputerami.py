import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
import os

class plansza:
    def __init__(self,szerokosc = 8, wysokosc = 10):
        self.tablica = np.repeat(0,(szerokosc+1) * (wysokosc +3)).reshape((szerokosc+1) , (wysokosc +3))
        self.tablica = self.tablica.astype(np.uint8)
        self.pozycja = np.array([0,0])
        self.srodek = np.array([(self.tablica.shape[0])/2, (self.tablica.shape[1])/2],dtype = 'uint8')
        self.pilka = self.srodek + self.pozycja
        self.indeks_pilki = (self.pilka[0],self.pilka[1])
        self.mapowanie_ruchow = {
                '4':  0,    # L
                '7':  1,   # LG
                '8':  2,    # G
                '9':  3,    # PG
                '6':  4,     # P
                '3':  5,     # PD
                '2':  6,     # D
                '1':  7,    # LD
                }
        
        self.mapowanie_odwrotne = {
                '0':  4,    # L
                '1':  7,   # LG
                '2':  8,    # G
                '3':  9,    # PG
                '4':  6,     # P
                '5':  3,     # PD
                '6':  2,     # D
                '7':  1,    # LD
                }
        
        self. przesuniecia = {
                '0':  np.array([0,-1]),    # L
                '1':  np.array([-1,-1]),   # LG
                '2':  np.array([-1,0]),    # G
                '3':  np.array([-1,1]),    # PG
                '4':  np.array([0,1]),     # P
                '5':  np.array([1,1]),     # PD
                '6':  np.array([1,0]),     # D
                '7':  np.array([1,-1]),    # LD
                }
        self. ruchy_odwrotne = {
                '0':  4,   # L -> P
                '1':  5,   # LG -> PD
                '2':  6,   # G -> D
                '3':  7,   # PG -> LD
                '4':  0,   # P -> L
                '5':  1,   # PD -> LG
                '6':  2,   # D -> G
                '7':  3,   # LD -> PG
                }
        
        self.x_ramka = [1,1,0,0,1,1,11,11,12,12,11,11,1]
        self.y_ramka = [0,3,3,5,5,8,8,5,5,3,3,0,0]
        self.x_trasa = [self.indeks_pilki[1]]
        self.y_trasa = [self.indeks_pilki[0]]
        
        self.x_ramka_symetryczna = [1,1,0,0,1,1,11,11,12,12,11,11,1]
        self.y_ramka_symetryczna = [0,3,3,5,5,8,8,5,5,3,3,0,0]
        self.x_trasa_symetryczna = [self.indeks_pilki[1]]
        self.y_trasa_symetryczna = [self.indeks_pilki[0]]
        
        self.specjalne_indeksy = dict()
        self.wyklucznie_ruchow()      
        
        self.tablica_symetryczna = np.copy(np.rot90(self.tablica,2))
        
        self.pozycja_symetryczna = np.array([0,0])
        self.pilka_symetryczna = self.srodek + self.pozycja
        self.indeks_pilki_symetryczny = (self.pilka[0],self.pilka[1])
        
    def mozliwe_ruchy(self):
        negacja = np.binary_repr(np.invert(self.tablica[self.indeks_pilki]),width = 8)
        ruchy = []
        pozycja = 7
        for i in negacja:
            if i == '1':
                ruchy.append(pozycja)
            pozycja = pozycja - 1
        
        return(ruchy)
        
    def mozliwe_ruchy_klawiatura(self):
        ruchy = self.mozliwe_ruchy()
        ruchy_klawiatura = []
        for ruch in ruchy:
            ruchy_klawiatura.append(self.mapowanie_odwrotne[str(ruch)])
        
        print(ruchy_klawiatura)
    
    def przesun_pilke(self,ruch):
        przesuniecie = self.przesuniecia[str(ruch)]
        self.pozycja = self.pozycja + przesuniecie
        self.pilka = self.srodek + self.pozycja
        self.indeks_pilki = (self.pilka[0],self.pilka[1])  
        ruch_odwrotny  = self.ruchy_odwrotne[str(ruch)]
        nowa_wartosc = np.bitwise_or(self.tablica[self.indeks_pilki], 2**ruch_odwrotny)
        kolejny_ruch = self.czy_kolejny_ruch()
        #if self.tablica[self.indeks_pilki] > 0:
        #    print('Kolejny ruch')
        #    kolejny_ruch = True
        self.tablica[self.indeks_pilki] = nowa_wartosc
        self.x_trasa.append(self.indeks_pilki[1])
        self.y_trasa.append(8 - self.indeks_pilki[0])
        return(kolejny_ruch)
        
    def wykonaj_ruch(self,ruch):
        #if np.binary_and(2**ruch, self.tablica[self.indeks_pilki]) > 0:
        #    print('Blad')
        nowa_wartosc = np.bitwise_or(self.tablica[self.indeks_pilki], 2**ruch)
        self.tablica[self.indeks_pilki] = nowa_wartosc
        kolejny_ruch = self.przesun_pilke(ruch)  
        nagroda = 0
        if self.indeks_pilki[1] == 12:
            nagroda = 1
        if self.indeks_pilki[1] == 0:
            nagroda = -1
        
        return (kolejny_ruch,nagroda)
    
    def ruch_z_klawiatury(self,wejscie):
        ruch = self.mapowanie_ruchow[wejscie]
        return(self.wykonaj_ruch(ruch))
        #self.wykonaj_ruch_symetryczny(ruch)
        
    def wyswietl(self):
        plt.plot(self.x_ramka,self.y_ramka, c = 'k')
        plt.plot(self.x_trasa,self.y_trasa, c = 'g')
        plt.scatter([self.indeks_pilki[1]],[8 - self.indeks_pilki[0]], c = 'r')
        plt.show()
        
    def czy_kolejny_ruch(self):
        if self.tablica[self.indeks_pilki] == 0:
            return(False)
        else:
            kolejny_ruch = True
            if str(self.indeks_pilki) in self.specjalne_indeksy:
                if self.specjalne_indeksy[str(self.indeks_pilki)] == self.tablica[self.indeks_pilki]:
                    kolejny_ruch = False
            
            return(kolejny_ruch)
                
        
    def wyswietl_symetrycznie(self):
        plt.plot(self.x_ramka_symetryczna,self.y_ramka_symetryczna, c = 'k')
        plt.plot(self.x_trasa_symetryczna,self.y_trasa_symetryczna, c = 'g')
        plt.scatter([self.indeks_pilki_symetryczny[1]],[8 - self.indeks_pilki_symetryczny[0]], c = 'r')
        plt.show()
    
    
    def wykonaj_ruch_symetryczny(self,ruch):
        #if np.binary_and(2**ruch, self.tablica[self.indeks_pilki]) > 0:
        #    print('Blad')
        ruch_odwrotny  = self.ruchy_odwrotne[str(ruch)]
        nowa_wartosc = np.bitwise_or(self.tablica[self.indeks_pilki], 2**ruch_odwrotny)
        self.tablica_symetryczna[self.indeks_pilki_symetryczny] = nowa_wartosc
        self.przesun_pilke_symetrycznie(ruch_odwrotny)  
        
    def przesun_pilke_symetrycznie(self,ruch):
        przesuniecie = self.przesuniecia[str(ruch)]
        self.pozycja_symetryczna = self.pozycja_symetryczna + przesuniecie
        self.pilka_symetryczna = self.srodek + self.pozycja_symetryczna
        self.indeks_pilki_symetryczny = (self.pilka_symetryczna[0],self.pilka_symetryczna[1])  
        ruch_odwrotny  = self.ruchy_odwrotne[str(ruch)]
        nowa_wartosc = np.bitwise_or(self.tablica_symetryczna[self.indeks_pilki_symetryczny], 2**ruch_odwrotny)
        if self.tablica[self.indeks_pilki_symetryczny] > 0:
            print('Kolejny ruch')
        self.tablica_symetryczna[self.indeks_pilki_symetryczny] = nowa_wartosc
        self.x_trasa_symetryczna.append(self.indeks_pilki_symetryczny[1])
        self.y_trasa_symetryczna.append(8 - self.indeks_pilki_symetryczny[0])
        
    def wyklucznie_ruchow(self):
        rogi = [(0,1),(0,11), (8,1), (8,11),(3,0),(5,0),(3,12),(5,12)]
        for i in rogi:
            self.tablica[i] = 255
        
        # gorna scianka
        zabronione_ruchy = ['4','7','8','9','6']
        for i in range(2,11):
            for r in zabronione_ruchy:
                ruch = self.mapowanie_ruchow[r]
                nowa_wartosc = np.bitwise_or(self.tablica[(0,i)], 2**ruch)
                self.tablica[(0,i)] = nowa_wartosc
        
        # dolna scianka
        zabronione_ruchy = ['4','1','2','3','6']
        for i in range(2,11):
            for r in zabronione_ruchy:
                ruch = self.mapowanie_ruchow[r]
                nowa_wartosc = np.bitwise_or(self.tablica[(8,i)], 2**ruch)
                self.tablica[(8,i)] = nowa_wartosc
        
        wysokosci_scianek = [1,2,6,7]
        
        # lewa scianka
        zabronione_ruchy = ['4','1','2','7','8']
        for i in wysokosci_scianek:
            for r in zabronione_ruchy:
                ruch = self.mapowanie_ruchow[r]
                nowa_wartosc = np.bitwise_or(self.tablica[(i,1)], 2**ruch)
                self.tablica[(i,1)] = nowa_wartosc
        
        # prawa scianka
        zabronione_ruchy = ['8','9','6','3','2']
        for i in wysokosci_scianek:
            for r in zabronione_ruchy:
                ruch = self.mapowanie_ruchow[r]
                nowa_wartosc = np.bitwise_or(self.tablica[(i,11)], 2**ruch)
                self.tablica[(i,11)] = nowa_wartosc
        
        # rogi bramek
        # lewy dolny
        zabronione_ruchy = ['2','1','4']
        for r in zabronione_ruchy:
            ruch = self.mapowanie_ruchow[r]
            nowa_wartosc = np.bitwise_or(self.tablica[(5,1)], 2**ruch)
            self.tablica[(5,1)] = nowa_wartosc
        
        # lewy gorny
        zabronione_ruchy = ['4','7','8']
        for r in zabronione_ruchy:
            ruch = self.mapowanie_ruchow[r]
            nowa_wartosc = np.bitwise_or(self.tablica[(3,1)], 2**ruch)
            self.tablica[(3,1)] = nowa_wartosc
        
        # prawy dolny
        zabronione_ruchy = ['2','3','6']
        for r in zabronione_ruchy:
            ruch = self.mapowanie_ruchow[r]
            nowa_wartosc = np.bitwise_or(self.tablica[(5,11)], 2**ruch)
            self.tablica[(5,11)] = nowa_wartosc
        
        # prawy gorny
        zabronione_ruchy = ['6','9','8']
        for r in zabronione_ruchy:
            ruch = self.mapowanie_ruchow[r]
            nowa_wartosc = np.bitwise_or(self.tablica[(3,11)], 2**ruch)
            self.tablica[(3,11)] = nowa_wartosc
        
        # okolice rogow boiska
        # lewy dolny
        zabronione_ruchy = ['1']
        for r in zabronione_ruchy:
            ruch = self.mapowanie_ruchow[r]
            nowa_wartosc = np.bitwise_or(self.tablica[(7,2)], 2**ruch)
            self.tablica[(7,2)] = nowa_wartosc
            self.specjalne_indeksy[str((7,2))] = nowa_wartosc
        
        # lewy gorny
        zabronione_ruchy = ['7']
        for r in zabronione_ruchy:
            ruch = self.mapowanie_ruchow[r]
            nowa_wartosc = np.bitwise_or(self.tablica[(1,2)], 2**ruch)
            self.tablica[(1,2)] = nowa_wartosc
            self.specjalne_indeksy[str((1,2))] = nowa_wartosc
        
        # prawy dolny
        zabronione_ruchy = ['3']
        for r in zabronione_ruchy:
            ruch = self.mapowanie_ruchow[r]
            nowa_wartosc = np.bitwise_or(self.tablica[(7,10)], 2**ruch)
            self.tablica[(7,10)] = nowa_wartosc
            self.specjalne_indeksy[str((7,10))] = nowa_wartosc
        
        # prawy gorny
        zabronione_ruchy = ['9']
        for r in zabronione_ruchy:
            ruch = self.mapowanie_ruchow[r]
            nowa_wartosc = np.bitwise_or(self.tablica[(1,10)], 2**ruch)
            self.tablica[(1,10)] = nowa_wartosc
            self.specjalne_indeksy[str((1,10))] = nowa_wartosc
            
        # linie bramek
        # lewa:
        zabronione_ruchy = ['1','7']
        for r in zabronione_ruchy:
            ruch = self.mapowanie_ruchow[r]
            nowa_wartosc = np.bitwise_or(self.tablica[(4,1)], 2**ruch)
            self.tablica[(4,1)] = nowa_wartosc
            self.specjalne_indeksy[str((4,1))] = nowa_wartosc
        
        # prawa:
        zabronione_ruchy = ['3','9']
        for r in zabronione_ruchy:
            ruch = self.mapowanie_ruchow[r]
            nowa_wartosc = np.bitwise_or(self.tablica[(4,11)], 2**ruch)
            self.tablica[(4,11)] = nowa_wartosc
            self.specjalne_indeksy[str((4,11))] = nowa_wartosc

class CustomMSE(keras.losses.Loss):
    def __init__(self, wyjscie, name="custom_mse"):
        super().__init__(name=name)
        self.wyjscie = wyjscie

    def call(self, y_true, y_pred):
        mse = tf.math.reduce_mean(tf.square(y_true[0,self.wyjscie] - y_pred[0,self.wyjscie]))
        return mse

import tensorflow.keras.backend as kb
def custom_loss(i,y_actual,y_pred):
    custom_loss=kb.square(y_actual[0,i]-y_pred[0,i])
    return custom_loss

class gracz_lewy:
    def __init__(self,imie = 'Lewy'):
        self.imie = imie
        self.stany = []
        self.kolejne_stany = []
        self.nagrody = []
        self.wejscia = 101
        self.ruchy = []
    
    def stworz_siec(self):
        inputs = keras.Input(shape=(self.wejscia,), name="Wejscie")
        x = layers.Dense(64, activation="softsign", name="dense_1")(inputs)
        x = layers.Dense(64, activation="softsign", name="dense_2")(x)
        outputs = [layers.Dense(1, activation="softsign", name="predictions1")(x),
                   layers.Dense(1, activation="softsign", name="predictions2")(x),
                   layers.Dense(1, activation="softsign", name="predictions3")(x),
                   layers.Dense(1, activation="softsign", name="predictions4")(x),
                   layers.Dense(1, activation="softsign", name="predictions5")(x),
                   layers.Dense(1, activation="softsign", name="predictions6")(x),
                   layers.Dense(1, activation="softsign", name="predictions7")(x),
                   layers.Dense(1, activation="softsign", name="predictions8")(x)]

        self.model = keras.Model(inputs=inputs, outputs=outputs)
        
        self.model.compile(
                optimizer=tf.keras.optimizers.SGD(),
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=[tf.keras.metrics.MeanSquaredError()])
                
    
    def zobacz_ruchy(self,tablica):
        mozliwe_ruchy = tablica.mozliwe_ruchy()
        return(mozliwe_ruchy)
    
    def wykonaj_ruch(self,tablica):
        mozliwe_ruchy = tablica.mozliwe_ruchy()
        if len(mozliwe_ruchy) == 0:
            return(False,-1)
        else:
            ruch = self.wybierz_ruch(tablica,mozliwe_ruchy)
            return(tablica.wykonaj_ruch(ruch))
    
    def losowy_ruch(self,tablica):
        mozliwe_ruchy = tablica.mozliwe_ruchy()
        if len(mozliwe_ruchy) == 0:
            return(-1)
        else:
            ruch = mozliwe_ruchy[np.random.randint(0,len(mozliwe_ruchy))]
            return(ruch)
    
    def ruch_siec(self,tablica):
        stan = self.pobierz_stan(tablica)
        wynik = self.model.predict(stan.reshape(1,self.wejscia))
        wynik = np.array(wynik).flatten()
        print(wynik)
        mozliwe_ruchy = tablica.mozliwe_ruchy()
        for i in range(len(wynik)):
            if i not in mozliwe_ruchy:
                wynik[i] = np.NINF
        return(np.argmax(wynik))
        
    def aktualizuj_siec(self,x,y,pozycja):
        weights = [0.0] * 8
        weights[pozycja] = 1.0
        powtorzenia = 1
        if (y == -1) or (y == 1):
            powtorzenia = 3
        self.model.compile(
                 optimizer=tf.keras.optimizers.SGD(),
                 loss=tf.keras.losses.MeanSquaredError(),
                 metrics=[tf.keras.metrics.MeanSquaredError()],
                 loss_weights = weights)
        self.model.fit(
                x.reshape(1,self.wejscia),
                [np.array([y]).reshape(1,1).astype(np.float32, copy=False)] * 8,
                epochs=powtorzenia
                )
        
    
    
    def optymalna_nagroda(self,x):
        wynik = self.model.predict(x.reshape(1,self.wejscia))
        return(np.max(wynik))
    
    def wybierz_ruch(self,tablica,mozliwe_ruchy):
        return(mozliwe_ruchy[np.random.randint(0,len(mozliwe_ruchy))])
    
    def ruch_na_tablicy(self,tablica,ruch):
        return(tablica.wykonaj_ruch(ruch))
    
    def pobierz_stan(self,tablica):
        tab = (tablica.tablica[:,1:-1]/255).flatten()
        poz_y = tablica.pozycja[0]/4
        poz_x = tablica.pozycja[1]/5
        return(np.append(tab,[poz_y,poz_x]))
    
    def ruch(self,tablica,eps,gamma):
        stan = self.pobierz_stan(tablica)
        if len(tablica.mozliwe_ruchy()) == 0:
            nagroda = -1
            kolejny_ruch = False
        else:            
            if np.random.uniform() < eps:
                ruch = self.losowy_ruch(tablica)
            else:
                ruch = self.ruch_siec(tablica)
        
            kolejny_ruch,nagroda = self.ruch_na_tablicy(tablica,ruch)
            kolejny_stan = self.pobierz_stan(tablica)
            
        if nagroda != 0:
            y = nagroda
            ruch = self.ruchy[len(self.ruchy) -1]
            self.aktualizuj_siec(stan,y,ruch)
        else:
            self.stany.append(stan)
            self.kolejne_stany.append(kolejny_stan)
            self.nagrody.append(nagroda)
            self.ruchy.append(ruch)
            
            los = np.random.randint(0,len(self.stany))
            kolejny_stan_temp = self.kolejne_stany[los]
            y = self.nagrody[los] + gamma * self.optymalna_nagroda(kolejny_stan_temp)
            self.aktualizuj_siec(self.stany[los],y,self.ruchy[los])
        
        return((kolejny_ruch,nagroda))
        
    def resetuj(self):
        self.stany = []
        self.kolejne_stany = []
        self.nagrody = []
        self.ruchy = []
        
    def finalna_nagroda(self,nagroda):
        ostatni_ruch = self.ruchy[len(self.ruchy) - 1]
        ostatni_stan = self.stany[len(self.stany) - 1]
        self.aktualizuj_siec(ostatni_stan,nagroda,ostatni_ruch)  
    
    def wczytaj_model(self,sciezka):
        self.model = keras.models.load_model(sciezka)
        
    def wykonaj_najlepszy_ruch(self,tablica):
        if len(tablica.mozliwe_ruchy()) == 0:
            nagroda = -1
            kolejny_ruch = False
            return((kolejny_ruch,nagroda))
        else:
            ruch = self.ruch_siec(tablica)
            return(self.ruch_na_tablicy(tablica,ruch))
             
        

class gracz_prawy(gracz_lewy):
        
    def wykonaj_ruch(self,tablica):
        mozliwe_ruchy = tablica.mozliwe_ruchy()
        if len(mozliwe_ruchy) == 0:
            return(False,-1)
        else:
            ruch = self.wybierz_ruch(tablica,mozliwe_ruchy)
            do_zwrotu = tablica.wykonaj_ruch(ruch)
            return(do_zwrotu[0],-1 * do_zwrotu[1])
     
    def ruch_na_tablicy(self,tablica,ruch):
        wynik = tablica.wykonaj_ruch(ruch)
        return(wynik[0],-1 * wynik[1])

class gracz_czlowiek(gracz_lewy):
        
    def wykonaj_ruch(self,tablica):
        mozliwe_ruchy = tablica.mozliwe_ruchy()
        if len(mozliwe_ruchy) == 0:
            return(False,-1)
        else:
            ruch = input('Wykonaj ruch')
            do_zwrotu = tablica.ruch_z_klawiatury(ruch)
            return(do_zwrotu[0],-1 * do_zwrotu[1])
     
    def ruch_na_tablicy(self,tablica,ruch):
        wynik = tablica.wykonaj_ruch(ruch)
        return(wynik[0],-1 * wynik[1])   


p_1 = plansza()
g_L = gracz_lewy()
g_P = gracz_prawy('Prawy')

folder_gracz_lewy = 'model_gracz_lewy'
folder_gracz_prawy = 'model_gracz_prawy'

if os.path.exists('./' + folder_gracz_lewy) and os.path.exists('./' + folder_gracz_prawy):
    g_L.wczytaj_model(folder_gracz_lewy)
    g_P.wczytaj_model(folder_gracz_prawy)
else:
    g_L.stworz_siec()
    g_P.stworz_siec()
    g_L.model.save(folder_gracz_lewy)
    g_P.model.save(folder_gracz_prawy)
gracze = [g_P,g_L]

nastepna = True

while(nastepna):
    eps = 0
    p_1 = plansza()
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
