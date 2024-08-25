
# print("hallo")
#nasilsin
#by


#module->dökümam bir .py dökümani
#packege -> dosya -klasör

print(9+7)
print('hi')
print(5+6)
"print(u)"
"ders"
print("ders")

###############################################
# Sayılar (Numbers) ve Karakter Dizileri (Strings)
###############################################

print("Hello world")
print("Hello AI Era")

print(9)
9.2
type(9.2)
type("Mrb")

###############################################
# Atamalar ve Değişkenler (Assignments & Variables)
###############################################
a = 9
a

b = "hello ai era"
b

c = 10
a * c
a * 10
d = a - c

###############################################
# Virtual Environment (Sanal Ortam)  ve (Package Managment) Paket Yönetimi
###############################################

# Sanal ortamların listelenmesi:
# conda env list

# Sanal ortam oluşturma:
# conda create –n myenv

# Sanal ortamı aktif etme:
# conda activate myenv

# Yüklü paketlerin listelenmesi:
# conda list

# Paket yükleme:
# conda install numpy

# Aynı anda birden fazla paket yükleme:
# conda install numpy scipy pandas

# Paket silme:
# conda remove package_name

# Belirli bir versiyona göre paket yükleme:
# conda install numpy=1.20.1

# Paket yükseltme:
# conda upgrade numpy

# Tüm paketlerin yükseltilmesi:
# conda upgrade –all

# pip: pypi (python package index) paket yönetim aracı

# Paket yükleme:
# pip install pandas

# Paket yükleme versiyona göre:
# pip install pandas==1.2.1



------------------------------------------------------------------------------------------------

###############################################
# VERİ YAPILARI (DATA STRUCTURES)
###############################################
# - Veri Yapılarına Giriş ve Hızlı Özet
# - Sayılar (Numbers): int, float, complex
# - Karakter Dizileri (Strings): str
# - Boolean (TRUE-FALSE): bool
# - Liste (List)
# - Sözlük (Dictionary)
# - Demet (Tuple)
# - Set


###############################################
# Veri Yapılarına Giriş ve Hızlı Özet
##############################################

# Sayılar: integer
x = 46
type(x)

# Sayılar: float
x = 10.3
type(x)

# Sayılar: complex
x = 2j + 1
type(x)

# String
x = "Hello ai era"
type(x)

# Boolean
True
False
type(True)
5 == 4
3 == 2
1 == 1
type(3 == 2)

# Liste
x = ["btc", "eth", "xrp"]
type(x)

# Sözlük (dictionary)
x = {"name": "Peter", "Age": 36}
type(x)

# Tuple
x = ("python", "ml", "ds")
type(x)

# Set
x = {"python", "ml", "ds"}
type(x)

# Not: Liste, tuple, set ve dictionary veri yapıları aynı zamanda Python Collections (Arrays) olarak geçmektedir.


###############################################
# Sayılar (Numbers): int, float, complex
###############################################

a = 5
b = 10.5

a * 3
a / 7
a * b / 10
a ** 2

#######################
# Tipleri değiştirmek
#######################

int(b)
float(a)

int(a * b / 10)

c = a * b / 10

int(c)


###############################################
# Karakter Dizileri (Strings)
###############################################

print("John")
print('John')

"John"
name = "John"
name = 'John'

#######################
# Çok Satırlı Karakter Dizileri
#######################

"""Veri Yapıları: Hızlı Özet, 
Sayılar (Numbers): int, float, complex, 
Karakter Dizileri (Strings): str, 
List, Dictionary, Tuple, Set, 
Boolean (TRUE-FALSE): bool"""

long_str = """Veri Yapıları: Hızlı Özet, 
Sayılar (Numbers): int, float, complex, 
Karakter Dizileri (Strings): str, 
List, Dictionary, Tuple, Set, 
Boolean (TRUE-FALSE): bool"""

#######################
# Karakter Dizilerinin Elemanlarına Erişmek
#######################

name
name[0]
name[3]
name[2]

#######################
# Karakter Dizilerinde Slice İşlemi
#######################

name[0:2]
long_str[0:10]

#######################
# String İçerisinde Karakter Sorgulamak
#######################

long_str

"veri" in long_str

"Veri" in long_str

"bool" in long_str

###############################################
# String (Karakter Dizisi) Metodları
###############################################

dir(str)

#######################
# len
#######################

name = "john"
type(name)
type(len)

len(name)
len("vahitkeskin")
len("miuul")

#######################
# upper() & lower(): küçük-büyük dönüşümleri
#######################

"miuul".upper()
"MIUUL".lower()

# type(upper)
# type(upper())


#######################
# replace: karakter değiştirir
#######################

hi = "Hello AI Era"
hi.replace("l", "p")

#######################
# split: böler
#######################

"Hello AI Era".split()

#######################
# strip: kırpar
#######################

" ofofo ".strip()
"ofofo".strip("o")


#######################
# capitalize: ilk harfi büyütür
#######################

"foo".capitalize()

dir("foo")

"foo".startswith("f")

###############################################
# Liste (List)
###############################################

# - Değiştirilebilir
# - Sıralıdır. Index işlemleri yapılabilir.
# - Kapsayıcıdır.

notes = [1, 2, 3, 4]
type(notes)
names = ["a", "b", "v", "d"]
not_nam = [1, 2, 3, "a", "b", True, [1, 2, 3]]

not_nam[0]
not_nam[5]
not_nam[6]
not_nam[6][1]

type(not_nam[6])

type(not_nam[6][1])

notes[0] = 99

not_nam[0:4]


###############################################
# Liste Metodları (List Methods)
###############################################

dir(notes)

#######################
# len: builtin python fonksiyonu, boyut bilgisi.
#######################

len(notes)
len(not_nam)

#######################
# append: eleman ekler
#######################

notes
notes.append(100)

#######################
# pop: indexe göre siler
#######################

notes.pop(0)

#######################
# insert: indexe ekler
#######################

notes.insert(2, 99)


###############################################
# Sözlük (Dictonary)
###############################################

# Değiştirilebilir.
# Sırasız. (3.7 sonra sıralı.)
# Kapsayıcı.

# key-value

dictionary = {"REG": "Regression",
              "LOG": "Logistic Regression",
              "CART": "Classification and Reg"}

dictionary["REG"]


dictionary = {"REG": ["RMSE", 10],
              "LOG": ["MSE", 20],
              "CART": ["SSE", 30]}

dictionary = {"REG": 10,
              "LOG": 20,
              "CART": 30}

dictionary["CART"][1]

#######################
# Key Sorgulama
#######################

"YSA" in dictionary

#######################
# Key'e Göre Value'ya Erişmek
#######################

dictionary["REG"]
dictionary.get("REG")

#######################
# Value Değiştirmek
#######################

dictionary["REG"] = ["YSA", 10]

#######################
# Tüm Key'lere Erişmek
#######################
dictionary.keys()

#######################
# Tüm Value'lara Erişmek
#######################

dictionary.values()


#######################
# Tüm Çiftleri Tuple Halinde Listeye Çevirme
#######################

dictionary.items()

#######################
# Key-Value Değerini Güncellemek
#######################

dictionary.update({"REG": 11})

#######################
# Yeni Key-Value Eklemek
#######################

dictionary.update({"RF": 10})

###############################################
# Demet (Tuple)
###############################################

# - Değiştirilemez.
# - Sıralıdır.
# - Kapsayıcıdır.

t = ("john", "mark", 1, 2)
type(t)

t[0]
t[0:3]

t[0] = 99

t = list(t)
t[0] = 99
t = tuple(t)



###############################################
# Set
###############################################

# - Değiştirilebilir.
# - Sırasız + Eşsizdir.
# - Kapsayıcıdır.

#######################
# difference(): İki kümenin farkı
#######################

set1 = set([1, 3, 5])
set2 = set([1, 2, 3])

# set1'de olup set2'de olmayanlar.
set1.difference(set2)
set1 - set2

# set2'de olup set1'de olmayanlar.
set2.difference(set1)
set2 - set1

#######################
# symmetric_difference(): İki kümede de birbirlerine göre olmayanlar
#######################

set1.symmetric_difference(set2)
set2.symmetric_difference(set1)

#######################
# intersection(): İki kümenin kesişimi
#######################

set1 = set([1, 3, 5])
set2 = set([1, 2, 3])

set1.intersection(set2)
set2.intersection(set1)

set1 & set2


#######################
# union(): İki kümenin birleşimi
#######################

set1.union(set2)
set2.union(set1)


#######################
# isdisjoint(): İki kümenin kesişimi boş mu?
#######################

set1 = set([7, 8, 9])
set2 = set([5, 6, 7, 8, 9, 10])

set1.isdisjoint(set2)
set2.isdisjoint(set1)


#######################
# isdisjoint(): Bir küme diğer kümenin alt kümesi mi?
#######################

set1.issubset(set2)
set2.issubset(set1)

#######################
# issuperset(): Bir küme diğer kümeyi kapsıyor mu?
#######################

set2.issuperset(set1)
set1.issuperset(set2)




---------------------------------------------------------------

###############################################
# FONKSİYONLAR, KOŞULLAR, DÖNGÜLER, COMPREHENSIONS
###############################################
# - Fonksiyonlar (Functions)
# - Koşullar (Conditions)
# - Döngüler (Loops)
# - Comprehesions


###############################################
# FONKSİYONLAR (FUNCTIONS)
###############################################

#######################
# Fonksiyon Okuryazarlığı
#######################

print("a", "b")

print("a", "b", sep="__")


#######################
# Fonksiyon Tanımlama
#######################

def calculate(x):
    print(x * 2)


calculate(5)


# İki argümanlı/parametreli bir fonksiyon tanımlayalım.

def summer(arg1, arg2):
    print(arg1 + arg2)


summer(7, 8)

summer(8, 7)

summer(arg2=8, arg1=7)


#######################
# Docstring
#######################

def summer(arg1, arg2):
    print(arg1 + arg2)


def summer(arg1, arg2):
    """
    Sum of two numbers

    Args:
        arg1: int, float
        arg2: int, float

    Returns:
        int, float

    """

    print(arg1 + arg2)


summer(1, 3)


#######################
# Fonksiyonların Statement/Body Bölümü
#######################

# def function_name(parameters/arguments):
#     statements (function body)

def say_hi():
    print("Merhaba")
    print("Hi")
    print("Hello")


say_hi()


def say_hi(string):
    print(string)
    print("Hi")
    print("Hello")


say_hi("miuul")


def multiplication(a, b):
    c = a * b
    print(c)


multiplication(10, 9)

# girilen değerleri bir liste içinde saklayacak fonksiyon.

list_store = []


def add_element(a, b):
    c = a * b
    list_store.append(c)
    print(list_store)


add_element(1, 8)
add_element(18, 8)
add_element(180, 10)


#######################
# Ön Tanımlı Argümanlar/Parametreler (Default Parameters/Arguments)
#######################

def divide(a, b):
    print(a / b)


divide(1, 2)


def divide(a, b):
    print(a / b)


divide(10)


def say_hi(string="Merhaba"):
    print(string)
    print("Hi")
    print("Hello")


say_hi("mrb")

#######################
# Ne Zaman Fonksiyon Yazma İhtiyacımız Olur?
#######################

# varm, moisture, charge

(56 + 15) / 80
(17 + 45) / 70
(52 + 45) / 80


# DRY

def calculate(varm, moisture, charge):
    print((varm + moisture) / charge)


calculate(98, 12, 78)


#######################
# Return: Fonksiyon Çıktılarını Girdi Olarak Kullanmak
######################

def calculate(varm, moisture, charge):
    print((varm + moisture) / charge)


# calculate(98, 12, 78) * 10

def calculate(varm, moisture, charge):
    return (varm + moisture) / charge


calculate(98, 12, 78) * 10

a = calculate(98, 12, 78)


def calculate(varm, moisture, charge):
    varm = varm * 2
    moisture = moisture * 2
    charge = charge * 2
    output = (varm + moisture) / charge
    return varm, moisture, charge, output


type(calculate(98, 12, 78))

varma, moisturea, chargea, outputa = calculate(98, 12, 78)


#######################
# Fonksiyon İçerisinden Fonksiyon Çağırmak
######################

def calculate(varm, moisture, charge):
    return int((varm + moisture) / charge)


calculate(90, 12, 12) * 10


def standardization(a, p):
    return a * 10 / 100 * p * p


standardization(45, 1)


def all_calculation(varm, moisture, charge, p):
    a = calculate(varm, moisture, charge)
    b = standardization(a, p)
    print(b * 10)


all_calculation(1, 3, 5, 12)


def all_calculation(varm, moisture, charge, a, p):
    print(calculate(varm, moisture, charge))
    b = standardization(a, p)
    print(b * 10)


all_calculation(1, 3, 5, 19, 12)

#######################
# Lokal & Global Değişkenler (Local & Global Variables)
#######################

list_store = [1, 2]

def add_element(a, b):
    c = a * b
    list_store.append(c)
    print(list_store)

add_element(1, 9)

###############################################
# KOŞULLAR (CONDITIONS)
###############################################

# True-False'u hatırlayalım.
1 == 1
1 == 2

# if
if 1 == 1:
    print("something")

if 1 == 2:
    print("something")

number = 11

if number == 10:
    print("number is 10")

number = 10
number = 20


def number_check(number):
    if number == 10:
        print("number is 10")

number_check(12)

#######################
# else
#######################

def number_check(number):
    if number == 10:
        print("number is 10")

number_check(12)


def number_check(number):
    if number == 10:
        print("number is 10")
    else:
        print("number is not 10")

number_check(12)

#######################
# elif
#######################

def number_check(number):
    if number > 10:
        print("greater than 10")
    elif number < 10:
        print("less than 10")
    else:
        print("equal to 10")

number_check(6)

###############################################
# DÖNGÜLER (LOOPS)
###############################################
# for loop

students = ["John", "Mark", "Venessa", "Mariam"]

students[0]
students[1]
students[2]

for student in students:
    print(student)

for student in students:
    print(student.upper())

salaries = [1000, 2000, 3000, 4000, 5000]

for salary in salaries:
    print(salary)


for salary in salaries:
    print(int(salary*20/100 + salary))

for salary in salaries:
    print(int(salary*30/100 + salary))

for salary in salaries:
    print(int(salary*50/100 + salary))

def new_salary(salary, rate):
    return int(salary*rate/100 + salary)

new_salary(1500, 10)
new_salary(2000, 20)

for salary in salaries:
    print(new_salary(salary, 20))


salaries2 = [10700, 25000, 30400, 40300, 50200]

for salary in salaries2:
    print(new_salary(salary, 15))

for salary in salaries:
    if salary >= 3000:
        print(new_salary(salary, 10))
    else:
        print(new_salary(salary, 20))



#######################
# Uygulama - Mülakat Sorusu
#######################

# Amaç: Aşağıdaki şekilde string değiştiren fonksiyon yazmak istiyoruz.

# before: "hi my name is john and i am learning python"
# after: "Hi mY NaMe iS JoHn aNd i aM LeArNiNg pYtHoN"

range(len("miuul"))
range(0, 5)

for i in range(len("miuul")):
    print(i)

# 4 % 2 == 0
# m = "miuul"
# m[0]

def alternating(string):
    new_string = ""
    # girilen string'in index'lerinde gez.
    for string_index in range(len(string)):
        # index çift ise büyük harfe çevir.
        if string_index % 2 == 0:
            new_string += string[string_index].upper()
        # index tek ise küçük harfe çevir.
        else:
            new_string += string[string_index].lower()
    print(new_string)

alternating("miuul")

#######################
# break & continue & while
#######################

salaries = [1000, 2000, 3000, 4000, 5000]

for salary in salaries:
    if salary == 3000:
        break
    print(salary)


for salary in salaries:
    if salary == 3000:
        continue
    print(salary)


# while

number = 1
while number < 5:
    print(number)
    number += 1

#######################
# Enumerate: Otomatik Counter/Indexer ile for loop
#######################

students = ["John", "Mark", "Venessa", "Mariam"]

for student in students:
    print(student)

for index, student in enumerate(students):
    print(index, student)

A = []
B = []

for index, student in enumerate(students):
    if index % 2 == 0:
        A.append(student)
    else:
        B.append(student)


#######################
# Uygulama - Mülakat Sorusu
#######################
# divide_students fonksiyonu yazınız.
# Çift indexte yer alan öğrencileri bir listeye alınız.
# Tek indexte yer alan öğrencileri başka bir listeye alınız.
# Fakat bu iki liste tek bir liste olarak return olsun.

students = ["John", "Mark", "Venessa", "Mariam"]


def divide_students(students):
    groups = [[], []]
    for index, student in enumerate(students):
        if index % 2 == 0:
            groups[0].append(student)
        else:
            groups[1].append(student)
    print(groups)
    return groups

st = divide_students(students)
st[0]
st[1]


#######################
# alternating fonksiyonunun enumerate ile yazılması
#######################

def alternating_with_enumerate(string):
    new_string = ""
    for i, letter in enumerate(string):
        if i % 2 == 0:
            new_string += letter.upper()
        else:
            new_string += letter.lower()
    print(new_string)

alternating_with_enumerate("hi my name is john and i am learning python")

#######################
# Zip
#######################

students = ["John", "Mark", "Venessa", "Mariam"]

departments = ["mathematics", "statistics", "physics", "astronomy"]

ages = [23, 30, 26, 22]

list(zip(students, departments, ages))

###############################################
# lambda, map, filter, reduce
###############################################

def summer(a, b):
    return a + b

summer(1, 3) * 9

new_sum = lambda a, b: a + b

new_sum(4, 5)

# map
salaries = [1000, 2000, 3000, 4000, 5000]

def new_salary(x):
    return x * 20 / 100 + x

new_salary(5000)

for salary in salaries:
    print(new_salary(salary))

list(map(new_salary, salaries))


# del new_sum
list(map(lambda x: x * 20 / 100 + x, salaries))
list(map(lambda x: x ** 2 , salaries))

# FILTER
list_store = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
list(filter(lambda x: x % 2 == 0, list_store))

# REDUCE
from functools import reduce
list_store = [1, 2, 3, 4]
reduce(lambda a, b: a + b, list_store)

###############################################
# COMPREHENSIONS
###############################################

#######################
# List Comprehension
#######################

salaries = [1000, 2000, 3000, 4000, 5000]

def new_salary(x):
    return x * 20 / 100 + x

for salary in salaries:
    print(new_salary(salary))

null_list = []

for salary in salaries:
    null_list.append(new_salary(salary))

null_list = []

for salary in salaries:
    if salary > 3000:
        null_list.append(new_salary(salary))
    else:
        null_list.append(new_salary(salary * 2))

[new_salary(salary * 2) if salary < 3000 else new_salary(salary) for salary in salaries]
[salary * 2 for salary in salaries]
[salary * 2 for salary in salaries if salary < 3000]
[salary * 2 if salary < 3000 else salary * 0 for salary in salaries]
[new_salary(salary * 2) if salary < 3000 else new_salary(salary * 0.2) for salary in salaries]
students = ["John", "Mark", "Venessa", "Mariam"]
students_no = ["John", "Venessa"]


[student.lower() if student in students_no else student.upper() for student in students]
[student.upper() if student not in students_no else student.lower() for student in students]

#######################
# Dict Comprehension
#######################

dictionary = {'a': 1,
              'b': 2,
              'c': 3,
              'd': 4}

dictionary.keys()
dictionary.values()
dictionary.items()

{k: v ** 2 for (k, v) in dictionary.items()}

{k.upper(): v for (k, v) in dictionary.items()}

{k.upper(): v*2 for (k, v) in dictionary.items()}



#######################
# Uygulama - Mülakat Sorusu
#######################

# Amaç: çift sayıların karesi alınarak bir sözlüğe eklenmek istemektedir
# Key'ler orjinal değerler value'lar ise değiştirilmiş değerler olacak.


numbers = range(10)
new_dict = {}

for n in numbers:
    if n % 2 == 0:
        new_dict[n] = n ** 2

{n: n ** 2 for n in numbers if n % 2 == 0}

#######################
# List & Dict Comprehension Uygulamalar
#######################

#######################
# Bir Veri Setindeki Değişken İsimlerini Değiştirmek
#######################

# before:
# ['total', 'speeding', 'alcohol', 'not_distracted', 'no_previous', 'ins_premium', 'ins_losses', 'abbrev']

# after:
# ['TOTAL', 'SPEEDING', 'ALCOHOL', 'NOT_DISTRACTED', 'NO_PREVIOUS', 'INS_PREMIUM', 'INS_LOSSES', 'ABBREV']

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

for col in df.columns:
    print(col.upper())


A = []

for col in df.columns:
    A.append(col.upper())

df.columns = A

df = sns.load_dataset("car_crashes")

df.columns = [col.upper() for col in df.columns]

#######################
# İsminde "INS" olan değişkenlerin başına FLAG diğerlerine NO_FLAG eklemek istiyoruz.
#######################

# before:
# ['TOTAL',
# 'SPEEDING',
# 'ALCOHOL',
# 'NOT_DISTRACTED',
# 'NO_PREVIOUS',
# 'INS_PREMIUM',
# 'INS_LOSSES',
# 'ABBREV']

# after:
# ['NO_FLAG_TOTAL',
#  'NO_FLAG_SPEEDING',
#  'NO_FLAG_ALCOHOL',
#  'NO_FLAG_NOT_DISTRACTED',
#  'NO_FLAG_NO_PREVIOUS',
#  'FLAG_INS_PREMIUM',
#  'FLAG_INS_LOSSES',
#  'NO_FLAG_ABBREV']

[col for col in df.columns if "INS" in col]

["FLAG_" + col for col in df.columns if "INS" in col]

["FLAG_" + col if "INS" in col else "NO_FLAG_" + col for col in df.columns]

df.columns = ["FLAG_" + col if "INS" in col else "NO_FLAG_" + col for col in df.columns]

#######################
# Amaç key'i string, value'su aşağıdaki gibi bir liste olan sözlük oluşturmak.
# Sadece sayısal değişkenler için yapmak istiyoruz.
#######################

# Output:
# {'total': ['mean', 'min', 'max', 'var'],
#  'speeding': ['mean', 'min', 'max', 'var'],
#  'alcohol': ['mean', 'min', 'max', 'var'],
#  'not_distracted': ['mean', 'min', 'max', 'var'],
#  'no_previous': ['mean', 'min', 'max', 'var'],
#  'ins_premium': ['mean', 'min', 'max', 'var'],
#  'ins_losses': ['mean', 'min', 'max', 'var']}

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

num_cols = [col for col in df.columns if df[col].dtype != "O"]
soz = {}
agg_list = ["mean", "min", "max", "sum"]

for col in num_cols:
    soz[col] = agg_list

# kısa yol
new_dict = {col: agg_list for col in num_cols}

df[num_cols].head()

df[num_cols].agg(new_dict)

---------------------------------------------
###############################################
# PYTHON İLE VERİ ANALİZİ (DATA ANALYSIS WITH PYTHON)
###############################################
# - NumPy
# - Pandas
# - Veri Görselleştirme: Matplotlib & Seaborn
# - Gelişmiş Fonksiyonel Keşifçi Veri Analizi (Advanced Functional Exploratory Data Analysis)

#############################################
# NUMPY
#############################################

# Neden NumPy? (Why Numpy?)
# NumPy Array'i Oluşturmak (Creating Numpy Arrays)
# NumPy Array Özellikleri (Attibutes of Numpy Arrays)
# Yeniden Şekillendirme (Reshaping)
# Index Seçimi (Index Selection)
# Slicing
# Fancy Index
# Numpy'da Koşullu İşlemler (Conditions on Numpy)
# Matematiksel İşlemler (Mathematical Operations)

#############################################
# Neden NumPy?
#############################################
import numpy as np
a = [1, 2, 3, 4]
b = [2, 3, 4, 5]

ab = []

for i in range(0, len(a)):
    ab.append(a[i] * b[i])

a = np.array([1, 2, 3, 4])
b = np.array([2, 3, 4, 5])
a * b




#############################################
# NumPy Array'i Oluşturmak (Creating Numpy Arrays)
#############################################
import numpy as np

np.array([1, 2, 3, 4, 5])
type(np.array([1, 2, 3, 4, 5]))
np.zeros(10, dtype=int)
np.random.randint(0, 10, size=10)
np.random.normal(10, 4, (3, 4))

#############################################
# NumPy Array Özellikleri (Attibutes of Numpy Arrays)
#############################################
import numpy as np

# ndim: boyut sayısı
# shape: boyut bilgisi
# size: toplam eleman sayısı
# dtype: array veri tipi

a = np.random.randint(10, size=5)
a.ndim
a.shape
a.size
a.dtype

#############################################
# Yeniden Şekillendirme (Reshaping)
#############################################
import numpy as np

np.random.randint(1, 10, size=9)
np.random.randint(1, 10, size=9).reshape(3, 3)

ar = np.random.randint(1, 10, size=9)
ar.reshape(3, 3)


#############################################
# Index Seçimi (Index Selection)
#############################################
import numpy as np
a = np.random.randint(10, size=10)
a[0]
a[0:5]
a[0] = 999

m = np.random.randint(10, size=(3, 5))

m[0, 0]
m[1, 1]
m[2, 3]

m[2, 3] = 999

m[2, 3] = 2.9

m[:, 0]
m[1, :]
m[0:2, 0:3]

#############################################
# Fancy Index
#############################################
import numpy as np

v = np.arange(0, 30, 3)
v[1]
v[4]

catch = [1, 2, 3]

v[catch]

#############################################
# Numpy'da Koşullu İşlemler (Conditions on Numpy)
#############################################
import numpy as np
v = np.array([1, 2, 3, 4, 5])

#######################
# Klasik döngü ile
#######################
ab = []
for i in v:
    if i < 3:
        ab.append(i)

#######################
# Numpy ile
#######################
v < 3

v[v < 3]
v[v > 3]
v[v != 3]
v[v == 3]
v[v >= 3]

#############################################
# Matematiksel İşlemler (Mathematical Operations)
#############################################
import numpy as np
v = np.array([1, 2, 3, 4, 5])

v / 5
v * 5 / 10
v ** 2
v - 1

np.subtract(v, 1)
np.add(v, 1)
np.mean(v)
np.sum(v)
np.min(v)
np.max(v)
np.var(v)
v = np.subtract(v, 1)

#######################
# NumPy ile İki Bilinmeyenli Denklem Çözümü
#######################

# 5*x0 + x1 = 12
# x0 + 3*x1 = 10

a = np.array([[5, 1], [1, 3]])
b = np.array([12, 10])

np.linalg.solve(a, b)

#############################################
# PANDAS
#############################################

# Pandas Series
# Veri Okuma (Reading Data)
# Veriye Hızlı Bakış (Quick Look at Data)
# Pandas'ta Seçim İşlemleri (Selection in Pandas)
# Toplulaştırma ve Gruplama (Aggregation & Grouping)
# Apply ve Lambda
# Birleştirme (Join) İşlemleri

#############################################
# Pandas Series
#############################################
import pandas as pd

s = pd.Series([10, 77, 12, 4, 5])
type(s)
s.index
s.dtype
s.size
s.ndim
s.values
type(s.values)
s.head(3)
s.tail(3)


#############################################
# Veri Okuma (Reading Data)
#############################################
import pandas as pd

df = pd.read_csv("datasets/advertising.csv")
df.head()
# pandas cheatsheet

#############################################
# Veriye Hızlı Bakış (Quick Look at Data)
#############################################
import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")
df.head()
df.tail()
df.shape
df.info()
df.columns
df.index
df.describe().T
df.isnull().values.any()
df.isnull().sum()
df["sex"].head()
df["sex"].value_counts()


#############################################
# Pandas'ta Seçim İşlemleri (Selection in Pandas)
#############################################
import pandas as pd
import seaborn as sns
df = sns.load_dataset("titanic")
df.head()

df.index
df[0:13]
df.drop(0, axis=0).head()

delete_indexes = [1, 3, 5, 7]
df.drop(delete_indexes, axis=0).head(10)

# df = df.drop(delete_indexes, axis=0)
# df.drop(delete_indexes, axis=0, inplace=True)

#######################
# Değişkeni Indexe Çevirmek
#######################

df["age"].head()
df.age.head()

df.index = df["age"]

df.drop("age", axis=1).head()

df.drop("age", axis=1, inplace=True)
df.head()

#######################
# Indexi Değişkene Çevirmek
#######################

df.index

df["age"] = df.index

df.head()
df.drop("age", axis=1, inplace=True)

df.reset_index().head()
df = df.reset_index()
df.head()

#######################
# Değişkenler Üzerinde İşlemler
#######################
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

"age" in df

df["age"].head()
df.age.head()

df["age"].head()
type(df["age"].head())


df[["age"]].head()
type(df[["age"]].head())

df[["age", "alive"]]

col_names = ["age", "adult_male", "alive"]
df[col_names]

df["age2"] = df["age"]**2
df["age3"] = df["age"] / df["age2"]

df.drop("age3", axis=1).head()

df.drop(col_names, axis=1).head()

df.loc[:, ~df.columns.str.contains("age")].head()


#######################
# iloc & loc
#######################
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

# iloc: integer based selection
df.iloc[0:3]
df.iloc[0, 0]

# loc: label based selection
df.loc[0:3]

df.iloc[0:3, 0:3]
df.loc[0:3, "age"]

col_names = ["age", "embarked", "alive"]
df.loc[0:3, col_names]


#######################
# Koşullu Seçim (Conditional Selection)
#######################
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

df[df["age"] > 50].head()
df[df["age"] > 50]["age"].count()

df.loc[df["age"] > 50, ["age", "class"]].head()

df.loc[(df["age"] > 50) & (df["sex"] == "male"), ["age", "class"]].head()

df["embark_town"].value_counts()

df_new = df.loc[(df["age"] > 50) & (df["sex"] == "male")
       & ((df["embark_town"] == "Cherbourg") | (df["embark_town"] == "Southampton")),
       ["age", "class", "embark_town"]]

df_new["embark_town"].value_counts()

#############################################
# Toplulaştırma ve Gruplama (Aggregation & Grouping)
#############################################

# - count()
# - first()
# - last()
# - mean()
# - median()
# - min()
# - max()
# - std()
# - var()
# - sum()
# - pivot table

import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

df["age"].mean()

df.groupby("sex")["age"].mean()

df.groupby("sex").agg({"age": "mean"})
df.groupby("sex").agg({"age": ["mean", "sum"]})

df.groupby("sex").agg({"age": ["mean", "sum"],
                       "survived": "mean"})


df.groupby(["sex", "embark_town"]).agg({"age": ["mean"],
                       "survived": "mean"})

df.groupby(["sex", "embark_town", "class"]).agg({"age": ["mean"],
                       "survived": "mean"})


df.groupby(["sex", "embark_town", "class"]).agg({
    "age": ["mean"],
    "survived": "mean",
    "sex": "count"})


#######################
# Pivot table
#######################
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

df.pivot_table("survived", "sex", "embarked")

df.pivot_table("survived", "sex", ["embarked", "class"])

df.head()

df["new_age"] = pd.cut(df["age"], [0, 10, 18, 25, 40, 90])

df.pivot_table("survived", "sex", ["new_age", "class"])

pd.set_option('display.width', 500)


#############################################
# Apply ve Lambda
#############################################
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

df["age2"] = df["age"]*2
df["age3"] = df["age"]*5

(df["age"]/10).head()
(df["age2"]/10).head()
(df["age3"]/10).head()

for col in df.columns:
    if "age" in col:
        print(col)

for col in df.columns:
    if "age" in col:
        print((df[col]/10).head())

for col in df.columns:
    if "age" in col:
        df[col] = df[col]/10

df.head()

df[["age", "age2", "age3"]].apply(lambda x: x/10).head()

df.loc[:, df.columns.str.contains("age")].apply(lambda x: x/10).head()

df.loc[:, df.columns.str.contains("age")].apply(lambda x: (x - x.mean()) / x.std()).head()

def standart_scaler(col_name):
    return (col_name - col_name.mean()) / col_name.std()

df.loc[:, df.columns.str.contains("age")].apply(standart_scaler).head()

# df.loc[:, ["age","age2","age3"]] = df.loc[:, df.columns.str.contains("age")].apply(standart_scaler)

df.loc[:, df.columns.str.contains("age")] = df.loc[:, df.columns.str.contains("age")].apply(standart_scaler)

df.head()

#############################################
# Birleştirme (Join) İşlemleri
#############################################
import numpy as np
import pandas as pd
m = np.random.randint(1, 30, size=(5, 3))
df1 = pd.DataFrame(m, columns=["var1", "var2", "var3"])
df2 = df1 + 99

pd.concat([df1, df2])

pd.concat([df1, df2], ignore_index=True)

#######################
# Merge ile Birleştirme İşlemleri
#######################

df1 = pd.DataFrame({'employees': ['john', 'dennis', 'mark', 'maria'],
                    'group': ['accounting', 'engineering', 'engineering', 'hr']})

df2 = pd.DataFrame({'employees': ['mark', 'john', 'dennis', 'maria'],
                    'start_date': [2010, 2009, 2014, 2019]})

pd.merge(df1, df2)
pd.merge(df1, df2, on="employees")

# Amaç: Her çalışanın müdürünün bilgisine erişmek istiyoruz.
df3 = pd.merge(df1, df2)

df4 = pd.DataFrame({'group': ['accounting', 'engineering', 'hr'],
                    'manager': ['Caner', 'Mustafa', 'Berkcan']})

pd.merge(df3, df4)


#############################################
# VERİ GÖRSELLEŞTİRME: MATPLOTLIB & SEABORN
#############################################

#############################################
# MATPLOTLIB
#############################################

# Kategorik değişken: sütun grafik. countplot bar
# Sayısal değişken: hist, boxplot


#############################################
# Kategorik Değişken Görselleştirme
#############################################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

df['sex'].value_counts().plot(kind='bar')
plt.show()

#############################################
# Sayısal Değişken Görselleştirme
#############################################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

plt.hist(df["age"])
plt.show()

plt.boxplot(df["fare"])
plt.show()

#############################################
# Matplotlib'in Özellikleri
#############################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

#######################
# plot
#######################

x = np.array([1, 8])
y = np.array([0, 150])

plt.plot(x, y)
plt.show()

plt.plot(x, y, 'o')
plt.show()

x = np.array([2, 4, 6, 8, 10])
y = np.array([1, 3, 5, 7, 9])

plt.plot(x, y)
plt.show()

plt.plot(x, y, 'o')
plt.show()



#######################
# marker
#######################

y = np.array([13, 28, 11, 100])

plt.plot(y, marker='o')
plt.show()

plt.plot(y, marker='*')
plt.show()

markers = ['o', '*', '.', ',', 'x', 'X', '+', 'P', 's', 'D', 'd', 'p', 'H', 'h']

#######################
# line
#######################

y = np.array([13, 28, 11, 100])
plt.plot(y, linestyle="dashdot", color="r")
plt.show()

#######################
# Multiple Lines
#######################

x = np.array([23, 18, 31, 10])
y = np.array([13, 28, 11, 100])
plt.plot(x)
plt.plot(y)
plt.show()

#######################
# Labels
#######################

x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.plot(x, y)
# Başlık
plt.title("Bu ana başlık")

# X eksenini isimlendirme
plt.xlabel("X ekseni isimlendirmesi")

plt.ylabel("Y ekseni isimlendirmesi")

plt.grid()
plt.show()

#######################
# Subplots
#######################

# plot 1
x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.subplot(1, 2, 1)
plt.title("1")
plt.plot(x, y)

# plot 2
x = np.array([8, 8, 9, 9, 10, 15, 11, 15, 12, 15])
y = np.array([24, 20, 26, 27, 280, 29, 30, 30, 30, 30])
plt.subplot(1, 2, 2)
plt.title("2")
plt.plot(x, y)
plt.show()


# 3 grafiği bir satır 3 sütun olarak konumlamak.
# plot 1
x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.subplot(1, 3, 1)
plt.title("1")
plt.plot(x, y)

# plot 2
x = np.array([8, 8, 9, 9, 10, 15, 11, 15, 12, 15])
y = np.array([24, 20, 26, 27, 280, 29, 30, 30, 30, 30])
plt.subplot(1, 3, 2)
plt.title("2")
plt.plot(x, y)

# plot 3
x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.subplot(1, 3, 3)
plt.title("3")
plt.plot(x, y)

plt.show()









#############################################
# SEABORN
#############################################
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
df = sns.load_dataset("tips")
df.head()

df["sex"].value_counts()
sns.countplot(x=df["sex"], data=df)
plt.show()

df['sex'].value_counts().plot(kind='bar')
plt.show()


#############################################
# Sayısal Değişken Görselleştirme
#############################################

sns.boxplot(x=df["total_bill"])
plt.show()

df["total_bill"].hist()
plt.show()


#############################################
# GELİŞMİŞ FONKSİYONEL KEŞİFÇİ VERİ ANALİZİ (ADVANCED FUNCTIONAL EDA)
#############################################
# 1. Genel Resim
# 2. Kategorik Değişken Analizi (Analysis of Categorical Variables)
# 3. Sayısal Değişken Analizi (Analysis of Numerical Variables)
# 4. Hedef Değişken Analizi (Analysis of Target Variable)
# 5. Korelasyon Analizi (Analysis of Correlation)


#############################################
# 1. Genel Resim
#############################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()
df.tail()
df.shape
df.info()
df.columns
df.index
df.describe().T
df.isnull().values.any()
df.isnull().sum()


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)

df = sns.load_dataset("flights")
check_df(df)


#############################################
# 2. Kategorik Değişken Analizi (Analysis of Categorical Variables)
#############################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

df["survived"].value_counts()
df["sex"].unique()
df["class"].nunique()

cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]

num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]

cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]

cat_cols = cat_cols + num_but_cat

cat_cols = [col for col in cat_cols if col not in cat_but_car]

df[cat_cols].nunique()

[col for col in df.columns if col not in cat_cols]


df["survived"].value_counts()
100 * df["survived"].value_counts() / len(df)

def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

cat_summary(df, "sex")

for col in cat_cols:
    cat_summary(df, col)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

cat_summary(df, "sex", plot=True)

for col in cat_cols:
    if df[col].dtypes == "bool":
        print("sdfsdfsdfsdfsdfsd")
    else:
        cat_summary(df, col, plot=True)


df["adult_male"].astype(int)


for col in cat_cols:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)
        cat_summary(df, col, plot=True)

    else:
        cat_summary(df, col, plot=True)




def cat_summary(dataframe, col_name, plot=False):

    if dataframe[col_name].dtypes == "bool":
        dataframe[col_name] = dataframe[col_name].astype(int)

        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("##########################################")

        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)
    else:
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("##########################################")

        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)

cat_summary(df, "adult_male", plot=True)





def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

cat_summary(df, "sex")



#############################################
# 3. Sayısal Değişken Analizi (Analysis of Numerical Variables)
#############################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()


cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]
cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]
cat_cols = cat_cols + num_but_cat
cat_cols = [col for col in cat_cols if col not in cat_but_car]



df[["age","fare"]].describe().T

num_cols = [col for col in df.columns if df[col].dtypes in ["int","float"]]
num_cols = [col for col in num_cols if col not in cat_cols]

def num_summary(dataframe, numerical_col):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

num_summary(df, "age")

for col in num_cols:
    num_summary(df, col)


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


num_summary(df, "age", plot=True)

for col in num_cols:
    num_summary(df, col, plot=True)


#############################################
# Değişkenlerin Yakalanması ve İşlemlerin Genelleştirilmesi
#############################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()
df.info()

# docstring

def grab_col_names(dataframe, cat_th=10,  car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi
    num_cols: list
        Numerik değişken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi

    Notes
    ------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < 10 and dataframe[col].dtypes in ["int", "float"]]

    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > 20 and str(dataframe[col].dtypes) in ["category", "object"]]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)


def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

cat_summary(df, "sex")

for col in cat_cols:
    cat_summary(df, col)



def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in num_cols:
    num_summary(df, col, plot=True)



# BONUS
df = sns.load_dataset("titanic")
df.info()

for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col, plot=True)


for col in num_cols:
    num_summary(df, col, plot=True)




#############################################
# 4. Hedef Değişken Analizi (Analysis of Target Variable)
#############################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")


for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

def grab_col_names(dataframe, cat_th=10,  car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi
    num_cols: list
        Numerik değişken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi

    Notes
    ------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]

    num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]

    cat_but_car = [col for col in df.columns if
                   df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

df.head()

df["survived"].value_counts()
cat_summary(df, "survived")

#######################
# Hedef Değişkenin Kategorik Değişkenler ile Analizi
#######################


df.groupby("sex")["survived"].mean()

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


target_summary_with_cat(df, "survived", "pclass")


for col in cat_cols:
    target_summary_with_cat(df, "survived", col)



#######################
# Hedef Değişkenin Sayısal Değişkenler ile Analizi
#######################


df.groupby("survived")["age"].mean()

df.groupby("survived").agg({"age":"mean"})

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


target_summary_with_num(df, "survived","age")

for col in num_cols:
    target_summary_with_num(df, "survived", col)



#############################################
# 5. Korelasyon Analizi (Analysis of Correlation)
#############################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = pd.read_csv("datasets/breast_cancer.csv")
df = df.iloc[:, 1:-1]
df.head()

num_cols = [col for col in df.columns if df[col].dtype in [int, float]]

corr = df[num_cols].corr()

sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()


#######################
# Yüksek Korelasyonlu Değişkenlerin Silinmesi
#######################

cor_matrix = df.corr().abs()


#           0         1         2         3
# 0  1.000000  0.117570  0.871754  0.817941
# 1  0.117570  1.000000  0.428440  0.366126
# 2  0.871754  0.428440  1.000000  0.962865
# 3  0.817941  0.366126  0.962865  1.000000


#     0        1         2         3
# 0 NaN  0.11757  0.871754  0.817941
# 1 NaN      NaN  0.428440  0.366126
# 2 NaN      NaN       NaN  0.962865
# 3 NaN      NaN       NaN       NaN


upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col]>0.90) ]
cor_matrix[drop_list]
df.drop(drop_list, axis=1)

def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list


high_correlated_cols(df)
drop_list = high_correlated_cols(df, plot=True)
df.drop(drop_list, axis=1)
high_correlated_cols(df.drop(drop_list, axis=1), plot=True)

# Yaklaşık 600 mb'lık 300'den fazla değişkenin olduğu bir veri setinde deneyelim.
# https://www.kaggle.com/c/ieee-fraud-detection/data?select=train_transaction.csv

df = pd.read_csv("datasets/fraud_train_transaction.csv")
len(df.columns)
df.head()

drop_list = high_correlated_cols(df, plot=True)

len(df.drop(drop_list, axis=1).columns)

type(adsa)



















































