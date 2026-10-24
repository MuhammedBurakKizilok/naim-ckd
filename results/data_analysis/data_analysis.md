# Veri analizi – UCI Chronic Kidney Disease

- **hasta_sayisi**: 400
- **ozellik_sayisi**: 24
- **sayisal**: 11
- **kategorik**: 13
- **ckd**: 250
- **notckd**: 150
- **toplam_eksik_hucre_orani_%**: 10.54
- **eksik_orani_ckd_%**: 15.13
- **eksik_orani_notckd_%**: 2.89
- **hic_eksigi_olmayan_hasta**: 158
- **hic_eksigi_olmayan_hasta_ckd**: 43
- **hic_eksigi_olmayan_hasta_notckd**: 115
- **hasta_basi_eksik_medyan**: 1.0
- **hasta_basi_eksik_max**: 11

## Değişkenler (eksiklik oranına göre sıralı, %)

| ozellik   | aciklama                                | tur       | degerler                              |   eksik_sayi |   eksik_oran |   eksik_ckd |   eksik_notckd |
|:----------|:----------------------------------------|:----------|:--------------------------------------|-------------:|-------------:|------------:|---------------:|
| rbc       | İdrarda kırmızı kan hücresi             | kategorik | normal / abnormal                     |          152 |         38   |        57.2 |            6   |
| rc        | Kırmızı kan hücresi sayısı (milyon/mm³) | sayısal   | 2.1 – 8 (medyan 4.8)                  |          131 |         32.8 |        49.6 |            4.7 |
| wc        | Beyaz kan hücresi sayısı (hücre/mm³)    | sayısal   | 2200 – 26400 (medyan 8000)            |          106 |         26.5 |        39.6 |            4.7 |
| pot       | Potasyum (mEq/L)                        | sayısal   | 2.5 – 47 (medyan 4.4)                 |           88 |         22   |        33.2 |            3.3 |
| sod       | Sodyum (mEq/L)                          | sayısal   | 4.5 – 163 (medyan 138)                |           87 |         21.8 |        32.8 |            3.3 |
| pcv       | Paketlenmiş hücre hacmi                 | sayısal   | 9 – 54 (medyan 40)                    |           71 |         17.8 |        26.8 |            2.7 |
| pc        | Püs hücresi                             | kategorik | normal / abnormal                     |           65 |         16.2 |        22.4 |            6   |
| hemo      | Hemoglobin (g)                          | sayısal   | 3.1 – 17.8 (medyan 12.65)             |           52 |         13   |        18.4 |            4   |
| su        | Şeker (idrar)                           | kategorik | 0 / 1 / 2 / 3 / 4 / 5                 |           49 |         12.2 |        17.6 |            3.3 |
| sg        | İdrar özgül ağırlığı                    | kategorik | 1.005 / 1.010 / 1.015 / 1.020 / 1.025 |           47 |         11.8 |        16.8 |            3.3 |
| al        | Albümin                                 | kategorik | 0 / 1 / 2 / 3 / 4 / 5                 |           46 |         11.5 |        16.4 |            3.3 |
| bgr       | Rastgele kan şekeri (mg/dl)             | sayısal   | 22 – 490 (medyan 121)                 |           44 |         11   |        15.2 |            4   |
| bu        | Kan üre (mg/dl)                         | sayısal   | 1.5 – 391 (medyan 42)                 |           19 |          4.8 |         5.2 |            4   |
| sc        | Serum kreatinin (mg/dl)                 | sayısal   | 0.4 – 76 (medyan 1.3)                 |           17 |          4.2 |         4.8 |            3.3 |
| bp        | Kan basıncı (mm/Hg)                     | sayısal   | 50 – 180 (medyan 80)                  |           12 |          3   |         4   |            1.3 |
| age       | Yaş (yıl)                               | sayısal   | 2 – 90 (medyan 55)                    |            9 |          2.2 |         3.2 |            0.7 |
| pcc       | Püs hücresi kümeleri                    | kategorik | present / notpresent                  |            4 |          1   |         0   |            2.7 |
| ba        | Bakteri                                 | kategorik | present / notpresent                  |            4 |          1   |         0   |            2.7 |
| htn       | Hipertansiyon                           | kategorik | yes / no                              |            2 |          0.5 |         0   |            1.3 |
| dm        | Diyabet                                 | kategorik | yes / no                              |            2 |          0.5 |         0   |            1.3 |
| cad       | Koroner arter hastalığı                 | kategorik | yes / no                              |            2 |          0.5 |         0   |            1.3 |
| appet     | İştah                                   | kategorik | good / poor                           |            1 |          0.2 |         0   |            0.7 |
| pe        | Ayakta ödem                             | kategorik | yes / no                              |            1 |          0.2 |         0   |            0.7 |
| ane       | Anemi                                   | kategorik | yes / no                              |            1 |          0.2 |         0   |            0.7 |

## Sabit ayrım (stratified, seed=42)

| set   |   n |   ckd |   notckd |   ckd_oran |   eksik_hucre_orani |
|:------|----:|------:|---------:|-----------:|--------------------:|
| train | 280 |   175 |      105 |       62.5 |                10.3 |
| val   |  60 |    37 |       23 |       61.7 |                11.5 |
| test  |  60 |    38 |       22 |       63.3 |                10.6 |
