
# coding: utf-8

# In[39]:


ERROR_THRESHOLD = 0.2
import numpy as np
import time
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import os
import json
import datetime
import pandas as pd
import re
factory = StemmerFactory()
stemmer = factory.create_stemmer()
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

def sigmoid_output_to_derivative(output):
    return output*(1-output)
 
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def think(sentence, show_details=False):
    x = bow(sentence.lower(), words, show_details)
    if show_details:
        print ("sentence:", sentence, "\n bow:", x)
    l0 = x
    l1 = sigmoid(np.dot(l0, synapse_0))
    l2 = sigmoid(np.dot(l1, synapse_1))
    return l2
synapse_file = 'model.json' 
with open(synapse_file) as data_file: 
    synapse = json.load(data_file) 
    synapse_0 = np.asarray(synapse['synapse0']) 
    synapse_1 = np.asarray(synapse['synapse1'])
    words = synapse['words']
    classes = synapse['classes']
def classify(sentence, show_details=False):
    results = think(sentence, show_details)

    results = [[i,r] for i,r in enumerate(results)] 
    results.sort(key=lambda x: x[1], reverse=True) 
    return_results =[[classes[r[0]],r[1]] for r in results]
    return_results=return_results[0]
    print ("%s \n classification: %s" % (sentence, return_results))


# In[40]:


#jokowi
classify("Ahok dimusuhi karena dianggap China, non-muslim &amp; keras.  @jokowi dimusuhi meskipun asli Jawa, sikapnya santun, muslim yg rajin sholat &amp; rajin puasa Senen-Kamis.  Ada apa dgn bangsa ini?  Disaat bangsa lain sibuk menciptakan teknologi, kita justru masih sibuk dg urusan agama dll")
classify("Tanggapan Warga Tidak Mampu Di Kabupaten Bogor Yg Berterima Kasih Kepada Jokowi Krn Menerima Aliran Listrik Gratis.Mana Ada era Konglomerat Pepo Aliran Listrik Gratis Buat Warga Nggak Mampu?. #JokowiMembangunIndonesia #SalamSatuJempol #JokowiLagi  #JokowiLagi #Jokowi1KaliLagi")
classify("Jokowi Konsisten, Terukur dan Tidak Suka Berwacana #01JokowiPresiden #01JokowiMenang #01IndonesiaMajuTerus")
classify("TGB juga menyebut Jokowi adalah contoh pemimpin yang selalu mencari tahu dan apa yang bisa segera dilakukan, Jokowi bukan pemimpin yang suka berwacana. #01JokowiPresiden #01JokowiMenang #01IndonesiaMajuTerus")
classify("Kita bangga terhadap POLRI yang bekerja bersama Pak @jokowi dengan penuh cinta dan kasih sayang.")
classify("Hari ini empat tahun yang lalu, untuk pertama kalinya saya bertemu Pak @jokowi dan bersepeda bareng di bund HI. Mudah2an bisa ketemu dan ngegowes lagi dengan beliau. Sehat terus ya Pak")
classify("Untuk perkara korupsi, Jokowi tidak pernah main-main. Hingga menjelang akhir periode kepemimpinanya sebagai presiden")
classify("Memang pembangunan yang telah kita kerjakan dalam empat tahun ini orientasinya kita ubah, kita balik. Pembangunan dimulai dari desa")
classify("Anggaran naik setiap tahunnya Jokowi mengatakan kenaikan alokasi anggaran untuk pembangunan desa setiap tahunnya menjadi salah satu indikasi, bahwa pemerintah memang memprioritaskan desa sebagai program utama pembangunan.")
classify("Hal tersebut ditegaskan Presiden Jokowi, bahwa orientasi pembangunan nasional memang beberapa tahun ini mengalami perubahan.")
classify("Ayoo  pilih Presiden yg berkualitas, di sukai kalangan bawah, menengah Dan atas, tahun depan jangan salah pilih yaa Jokowi lagi")
classify("Bangga punya pemimpin yang punya visi yang jelas, terukur, dan pasti. Pak Jokowi bersama jajarannya giat membangun infrastruktur dalam rangka membangun peradaban, konektivitas budaya, budaya baru")
classify("ayo diangkut avatarnya yahut Jokowi bikin salut buat dipilihnya patut kerjanya selalu dikebut uang negara tidak dicatut korupsi juga tidak prnah ikut👍 kluarga rukun &amp; jg penurut mentrinya brnama luhut mafia pd kalang kabut koruptor jadi pd ciut oposisi merengut yg  demo kalut ")
classify("Jokowi Minta Pembangunan Runway 3 Bandara Soetta Dipercepat")
classify("Di perkotaan juga menginginkan dana seperti Dana Desa Dalam perkembangannya, menurut Jokowi, perkotaan menginginkan kucuran dana serupa yang didapatkan desa.")
classify("Sepakat. Padahal jika kita ingin bersama membangun negeri, kita harus saling menghargai dan menghormati walau berbeda pendapat.")
classify("Pak Jokowi memang hebat. Luar biasa! Ga kebayang kalau misalnya yang menjabat presiden saat ini adalah Prabowo. Bisa dipastikan aksi itu bakalan rusuh dan berdarah-darah.")
classify("Sudah terbukti pak @jokowi tdk punya kemampuan membuat demo berjilid. Maklum, misi politik JKW: Kerja, kerja &amp; kerja. Hasilnya dinikmati ratusan juta rakyat. Jd mau demo ribuan kali pun sia2. Sbb rakyat akan memilih Capres yg kerja nyata, bukan Capres yg doyan bikin demo! ")
classify("Ini Presiden Gue Hari minggu tetap bekerja melayani Rakyat , tidak diundang ke acara Reuni tidak apa2 masih banyak tugas Negara yg Lebih penting membantu Masyarakat dg memberi penerangan kpd warga yg kurang mampu.")
classify("Part 1 : Presiden @jokowi : Pemilu Presiden, Pilih yang Punya Rekam Jejak dan Program Baik  https://t.co/SLck0PeHQ1  Saya yakin 2019 akan lebih dahsyat dari 2014. ")
classify("Dikasih pemimpin sprti Ahok protes, katanya Ahok kasar, kafir, china, bkn pribumi.  Dikasih pemimpin sprti Jokowi masih protes, ktanya gk bsa bhs. inggris, plonga-plongo, &amp; blm puas masih dfitnah PKI &amp; anti Islam.  Lucunya yg gk bisa ngaji malah dipilih utk dijadikan pemimpin?")
classify("Stabilitas politik dan keamanan itu sangat perlu dalam pembangunan kita, baik sekarang, jangka menengah, atau jangka panjang.  Maka, di acara Apel Danrem-Dandim Terpusat TA 2018 di Bandung, kemarin, saya kembali mengingatkan agar netralitas TNI terus dijaga.")
classify("👨‍🦱 Presiden kita kafir?  👳‍♂️Jangan ngaji disini, geli gw,,, sana kumpul sama yg sering triak2 di monas...  wkkkk keren ini Ustadz....")
classify("Ya Allah merdu sekali suara pak jokowi saat menyanyikan lagu ""jaenudin na ciro""...  Awas klo ada yg berani ketawa.. Gw tabok....!!! ")
classify("Hasrat yg terpendam 5 tahunan.. Awal nya @TitiekSoeharto malu malu meong. Akhir nya kikuk kikuk💏🤣🤣😂 Sikatt ommm @prabowo mumpung lgi nyapres,nnti klo habis nyapres ggal gk ada ksempatan lgi😛")
classify("Presiden Joko Widodo menyebutkan Sukabumi adalah wilayah yang memiliki potensi besar untuk perekonomian.")
classify("Jokowi menyebutkan, nantinya ruas tol yang diresmikan ini akan tersambung ke Cianjur, Bandung serta ke Cilacap, Jawa Tengah.")
classify("Selaras dg yg disampaikan Presiden, Dirut PT. PLN (Persero) Sofyan Basir mengatakan akan memasukkan seluruh KK tsb dlm program subsidi listrik dg pemasangan daya 450 VA.  Alhamdulillah ya bpk2 dan ibu2, semoga makin produktif dg ketersediaan listrik ini")
classify("Kerja Nyata Jokowi: Selesaikan Tol Bocimi, Setelah 21 Tahun Terbengkalai ")
classify("Jokowi: Ibu Jualannya Apa? Makemak: Saya Jualan Pisang Keju Jokowi: Lha Kok Persis Dengan Punyanya Anak Saya?!")
classify("Pak Jokowi memberikan bukti, infrastruktur Indonesia makin baik, rakyat makin cinta, semua siap bersatu")


# In[41]:


#prabowo
classify("Kalau melihat lautan manusia yang menghadiri Reuni Akbar 212, Pilpres 2019 sudah 'berakhir' bro, kecuali ada kecurangan.  Saya menyaksikan sendiri di lapangan begitu penuh sesaknya monas.  Umat semakin di dzalimi semakin bersatu padu &amp; kuat.")
classify("Jangan ada kecurangan PILPRES.. klw smpai ada kecurangan maka siap2 KPU akan di seruduk PULUHAN JUTA UMAT yg akan menuntut dan akan menggiring KPU ke MONAS untuk dimintai PERTANGGUNG JAWABANNYA.. MAKANYA KPU &amp; sypapun jgn coba2 tuk CURANGI hasil PILPRES 2019")
classify("Prabowo berdiri di atas kap mobil Lexus putih memberikan hormat balik kepada para prajurit Paspampres ketika melewati markas Paspampres di kawasan Jalan Tanah Abang, Jakarta Pusat.")
classify("Bung Denny, Reuni tadi itu bukan sedang membela Tuhan, tp sedang menjalankan perintah agama untuk menjaga silaturahmi.   Yg hadir td itu menjaga ukuwah persaudaraan dan persatuan untuk bersatu MENGGANTI PRESIDEN TAHUN DEPAN.   Tuhan tersenyum melihat umatnya rukun bersaudara. ")
classify("Jokowinya mah mungkin gak gelisah dan santai2 saja kan dia Boneka jadi apapun dia mah konsisten dg planga plongo nya, yg gelisah itu org2 yg di belakang Jokowi...!!!")
classify("Harus diakui, pak ji. Kalo bikin cerita2 bohong kayak gitu pendukung jokowi emang jagonya")
classify("Petani2 komoditi diluar Jawa, petani karet, kopra, sawit dll sedang betul2 terpukul. Kurs rupiah melemah seharusnya mereka diuntungkan spt 1998. Hari ini, kurs melemah tapi harga anjlok besar sekali. Harus ada kebijakan terobosan utk saudara2 kita petani komoditi")
classify("Jadi Gak Tega...😏 Pemilu semakin dekat kegagalan kepemimpinannya di buka dengan ""terang benderang"" gini..😭  Siapa Bilang Jokowi Berhasil di Infrastruktur? Ini Data-Data Kegagalannya")
classify("Buzzer Jokowow kepanasan. Mereka nga sadar aksi #2019GantiPresiden presiden sudah membesar. Masih aja mereka misuh2. Kekalahan Ahok makin menyimpan dendam. Kalau gue jadi kordinator mereka gue mandiin di got biar adem")
classify("Lalu kiai Makruf Amin berfatwa, jangan taati dan ikuti pemimpin yang ingkar janji. Maka Jkw dengan santun menjadikan  MA sebagai Wakilnya dalam rangka membantai otoritasnya sebagai Ulama, ketua MUI dan Rais Aam PBNU. Ini fakta nyata... CMIIW  #2019GantiPresiden ")
classify("Duh kepencet lagi 😜  .  Cebong makin meraung2 dah di tab mention .  ")
classify("Di pasar harga telor sudah 28rb jadi wajar kalo massa reoni 212 jauh lebih besar dari 2016. Karena gerakan Ini bukan lagi milik Umat Islam tapi juga milik kaum guru honorer, buruh, ojol &amp; rakyat Indonesia yang rindu akan perubahan!!   Cebong sesak nafas")
classify("Alhamdulillah, ini semua atas izin dan nikmat dari Allah subhanahu wa ta'ala.  Memang beda dengan tetangga sebelah, ngadain bagi sembako saja sampai ada yg meninggal karena pengaturan kegiatan yg buruk.")
classify("Ada yg bilang yg hadir reuni 212 itu KEBANYAKAN pendukung @prabowo , saya yakin itu SALAH. Bukan kebanyakan, tapi SEMUA pendukung Prabowo Sandi.  ")
classify("Mereka warga Nahdliyin yg sadar bahwa Kyai Ma'ruf tidak akan diberi peran signifikan spt pak JK. Hanya Vote gater .ide ekonomi keumatan takkan melaju kencang")
classify("Habib Rizieq Ajak Peserta Reuni 212 Dukung Prabowo-Sandiaga di Pilpres 2019 ")
classify("alam #ReuniAkbar212diMonas para Mujahid menyanyikan lagu Garuda Pancasila.  Nampak Antusias Peserta sambil bernyanyi, sambil mengibarkan bendera Tauhid &amp; Merah putih.  Moga ebong bs bobo mlm ini dg nyenyak stlh framingnya anti NKRI Gagal Total. ")
classify("Alhamdulillah, saya baru saja diberi kabar oleh Pak @prabowo bahwa acara Reuni 212 berlangsung dengan damai dan tertib sesuai dengan harapan kita semua. Banyak berkah yang didapat, ekonomi UMKM menggeliat, hotel-hotel juga penuh dan pariwisata DKI meningkat.")
classify("Jk gerakan 411, 212 sejak 2016 adalah gerakan makar spt mrk tuduhkan, mungkin sdh tdk ada kaum bernama Jokower krn junjungannya kemungkinan bsr sdh lengser.  Sesederhana itupun isi kepala mrk tak sanggup mencerna. ")
classify("Hari ini saya salah satu pribadi yang menjadi saksi betapa rukun, damai, dan tertibnya umat Islam serta para umat agama lain yang hadir dalam Reuni 212.")
classify("Yang saya khawatirkan adalah acara Jokowi di Istiqlal  Kita tau pendukungnya mayoritas tidak jelas agamanya  Saya khawatir kesucian Istiqlal ternoda. Sudahlah gak perlu bikin tandingan-tandingan  Jangan kotori Mesjid kami Jangan karena ingin 2 periode, mesjid kami tidak suci lagi")
classify("Dilantunkan dgn ringan tp penuh makna... Disampaikan dgn seni menusuk ke sanubari... Pesan tlah disampaikan, Makna tlah dipahami, Terpatri dlm sanubari....")
classify("@AnonBalle Brakakakakakka.... Ahlal kalam jd halal karam..  Yg lebih lucu jainudin naciro.. Presiden badut emang tugasnya bikin ketawa orang...")
classify("CEBONG CEBONG HARI INI PADA KEMANA YA???, 10 juta lebih dengan begitu damai, tertib, tidak ada yg rebutan nasi bungkus seperti masa yg dikumpulkan oleh kapitra dan orang2 bertopi lambang bintang, melihat aksi reuni 212 hari ini sdh tidak diragukan lagi 2019 pak @prabowo presiden")
classify("Presiden kerja itu yg ada kampanye terselubungnya")
classify("Bacain hestek alumni 212, jd ini sbnrnya acara buat menyatukan umat islam apa menyatukan org2 yg mau milih prabowo sandi? 😩")
classify("Jk gerakan 411, 212 sejak 2016 adalah gerakan makar spt mrk tuduhkan, mungkin sdh tdk ada kaum bernama Jokower krn junjungannya kemungkinan bsr sdh lengser.  Sesederhana itupun isi kepala mrk tak sanggup mencerna.")
classify("Hari ini saya salah satu pribadi yang menjadi saksi betapa rukun, damai, dan tertibnya umat Islam serta para umat agama lain yang hadir dalam Reuni 212.")
classify("Prabowo saat melakukan orasi di Reuni 212 di Monas. ''Di sini juga dihadiri tokoh agama lain. Ini bukti Islam di Indonesia adalah Islam yang damai dan mempersatukan!")
classify("Cebong jungkir balik mendiskreditkan, mendegradasi bahkan memfitnah aksi 212, kalau aksi ini pengaruhnya kecil knp kalian terlihat panik dan sewot, kalian klojotan smp air kolam terlihat keruh😂😂")


# In[ ]:


x = input()
classify(x,show_details=True)

