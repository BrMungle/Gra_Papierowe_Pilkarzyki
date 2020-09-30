# Uczenie ze wzmocnieniem zastosowane do gry w papierowe piłkarzyki

## O folderach...
W folderze 'nowe_poprawne_pliki' znajdują się pliki, które realizują to samo zadanie, co pliki w folderze 'stare_pliki', ale znacznie zgrabniej, wykorzystują znacznie mniej pamięci i pozwalają na wielokrotnie szybsze trenowanie sieci. Opiszę więc jedynie nowsze pliki, stare pewnie kiedyś stąd znikną, znajdują się tu jedynie dlatego, że te pliki właśnie znajdywały się w tym repozytorium w momencie, gdy składałem swoją pracę licencjacką, której fragment opisywał działanie umieszconego tu kodu. Stanowczo odradzam oglądanie zawartości folderu 'stare_pliki', wykorzystałem tam bibliotekę tensorflow bardzo nieumiejętnie.

## Folder nowe_poprawne_pliki
Krótkie opisy plików z tego folderu:
#### moje_klasy.py
W tym pliku zaimplementowane są klasy wykorzystane do implementacji gry w papierowe piłkarzyki oraz klasy reprezentujące graczy.
#### trenowanie_sieci.py
Plik, który realizuje trenowanie sieci neuronowych wykorzystywanych do sterowania każdym z graczy. W planach jest stworzenie nowych skryptów, które pozwolą na to, aby każdy z graczy korzystał z tej samej sieci, co może skutkować bardziej efektywnym trenowaniem sieci. Plik ten tworzy foldery, w których zapisywane są pliki reprezentujące sieci neuronowe. Dodatkowo wykorzystywany jest plik 'liczba_gier.txt' w celu zapamiętywania liczby dotychczas rozegranych gier. Ta liczba jest istotna z tego względu, że w zaimplementowanej metodzie trenowania wyznaczana jest przez nią szansa na wykonanie losowego ruchu przez agenta. 
#### gra_miedzy_botami.py
Skrypt ten ilustruje grę między stworzonymi modelami. Dobrze działa puszczony w środowisku Spyder. Daje możliwość porównywania ze sobą różnych modeli, stworzonych w różny sposób. 
#### gra_miedzy_botami.py
Skrypt ten umożliwia rozegranie gry przeciwko wybranemu botowi.
