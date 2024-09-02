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
#del dict[3]??????????????????


#Görev 5: Argüman olarak bir liste alan, listenin içerisindeki tek ve çift sayıları ayrı listelere atayan ve bu listeleri
#return eden fonksiyon yazınız.
