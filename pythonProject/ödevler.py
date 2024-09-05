#Görev 1: Verilen değerlerin veri yapılarını inceleyiniz.

x=8
type(x)

a='Hello World'
type(a)

l=[1,2,3,4]
type(l)

s={'pyt','ml','data Science'}
type(s)


#Görev 2: Verilen string ifadenin tüm harflerini büyük harfe çeviriniz. Virgül ve nokta yerine space koyunuz,
kelime kelime ayırınız.


text="The goal is to turn data info information,and information inti insight"



kelimeler = text.split()



büyük_harfli = [f"'{i.upper()}'" for i in kelimeler]


son = "["+', '.join(büyük_harfli)+"]"

print(son)


#Görev 3: Verilen listeye aşağıdaki adımları uygulayınız.

lst=["D","A","T","A","S","C","I","E","N","C","E"]

len(lst)

lst[0]
lst[10]

lst[0:4]

lst.pop(8)
lst

lst.append('x')
lst

lst.insert(8,'N')
lst


#Görev 4: Verilen sözlük yapısına aşağıdaki adımları uygulayınız.

dict={'Christian':['America',18],'Daisy':['England',12],'Antonio':['Spain',22],'Dante':['Italy',25]}
dict.keys()
dict.values()
dict['Daisy']=13
dict['Ahmet']=['Turkei,24']

dict.pop('Antonio')

dict

dict.pop('Antonio')







#Görev 5: Argüman olarak bir liste alan, listenin içerisindeki tek ve çift sayıları ayrı listelere atayan ve bu listeleri
#return eden fonksiyon yazınız.

l=[2,13,18,93,22]



A = []
B = []

for index, sayi in enumerate(l):
    if index % 2 == 0:
        A.append(sayi)
    else:
        B.append(sayi)
print(A)


vaya


def func(l):
    groups = [[], []]
    for index, sayi in enumerate(l):
        if index % 2 == 0:
            groups[0].append(sayi)
        else:
            groups[1].append(sayi)
    print(groups)
    return groups
func(l)


#Görev 6: Aşağıda verilen listede mühendislik ve tıp fakülterinde dereceye giren öğrencilerin isimleri
#bulunmaktadır. Sırasıyla ilk üç öğrenci mühendislik fakültesinin başarı sırasını temsil ederken son üç öğrenci de
#tıp fakültesi öğrenci sırasına aittir. Enumarate kullanarak öğrenci derecelerini fakülte özelinde yazdırınız.


ögrenciler=["Ali","Veli","Ayse","Talat","Zeynep","Ece"]


for i, ögrenci in enumerate(ögrenciler[:3], 1):
    print(f"Mühendislik Fakültesi {i}. öğrenci: {ögrenci}")

for i, ögrenci in enumerate(ögrenciler[3:], 1):
    print(f"Tıp Fakültesi {i}. öğrenci: {ögrenci}")


#Görev 7: Aşağıda 3 adet liste verilmiştir. Listelerde sırası ile bir dersin kodu, kredisi ve kontenjan bilgileri yer
#almaktadır. Zip kullanarak ders bilgilerini bastırınız.


ders_kodu=["Cmp105","Psy10001","Huk1005","Sen2204"]
kredi=[3,4,2,4]
kontenjan=[30,75,150,25]

list(zip(ders_kodu,kredi,kontenjan))



for a, b, c in zip(ders_kodu, kredi, kontenjan):
    print(f"Kredisi {a} olan {b} kodlu dersin kontenjanı {c} kişidir.")


#Görev 8: Aşağıda 2 adet set verilmiştir. Sizden istenilen eğer 1. küme 2. kümeyi kapsiyor ise ortak elemanlarını
#eğer kapsamıyor ise 2. kümenin 1. kümeden farkını yazdıracak fonksiyonu tanımlamanız beklenmektedir.

kume1=set(["data","python"])
kume2=set(["data","function","qcut","lambda","python","miuul"])


def kume_karsilastir(kume1, kume2):
    if kume1.issuperset(kume2):
        ortak_elemanlar = kume1.intersection(kume2)
        print(f"Ortak elemanlar: {ortak_elemanlar}")
    else:  # Eğer 1. küme, 2. kümeyi kapsamıyorsa
        fark = kume2.difference(kume1)
        print(f"2. kümenin 1. kümeden farkı: {fark}")


kume_karsilastir(kume1, kume2)

#List comprehension

#Görev 1: List Comprehension yapısı kullanarak car_crashes verisindeki numeric değişkenlerin isimlerini büyük
#harfe çeviriniz ve başına NUM ekleyiniz.

import seaborn as sns
df=sns.load_dataset("car_crashes")
df.columns

["NUM_" + col.upper() for col in df.columns]


#Görev 2: List Comprehension yapısı kullanarak car_crashes verisinde isminde "no" barındırmayan
#değişkenlerin isimlerinin sonuna "FLAG" yazınız.

["NUM_" + col.upper() for col in df.columns]
[col.upper()+"_FLAG"  if "no" not in col else col.upper() for col in df.columns ]


#Görev 3: List Comprehension yapısı kullanarak aşağıda verilen değişken isimlerinden FARKLI olan
#değişkenlerin isimlerini seçiniz ve yeni bir dataframe oluşturunuz.




import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

og_list=["abbrev","no_previous"]


new_cols = [col for col in df.columns if col not in og_list]

print(new_cols )

df[new_cols] .head(10)










[col for col in df.columns if df[col] != [og_list]]
soz = {}
agg_list = ["mean", "min", "max", "sum"]

for col in num_cols:
    soz[col] = agg_list

# kısa yol
new_dict = {col: agg_list for col in num_cols}

df[num_cols].head()

df[num_cols].agg(new_dict)
