import urllib.request


def  open_url(url):
    cookieProcessor = urllib.request.HTTPCookieProcessor()
    opener = urllib.request.build_opener(cookieProcessor)
    
    header={"User-Agent":r'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'}
    Cookies={"__cfduid":"d517f3a372e8b0b76ca65a5acd64ea6701528715770",
            "_ga":"GA1.2.283939496.1528715772", "_gid":"GA1.2.1079340513.1528715772",
            "euCookie":"set"}
    request=urllib.request.Request(url=url,headers=header)
    response = opener.open(request)
    return response.read()


def get_spectra(url, target_folder="/tmp/"):
    target = url.split("/")[-2]
    fname = target+".tar.gz"
    target_url = url+fname
    with open(target_folder + fname, 'wb') as fp:
        contents=open_url(target_url)
        fp.write(contents)
    print(f"Saved {fname:s}")
    

html = open_url("https://nwpsaf.eu/downloads/rtcoef_rttov12/ir_srf/")
from bs4 import BeautifulSoup

html_soup = BeautifulSoup(html, 'html.parser')
anchors = html_soup.find_all('a')

for anchor in anchors:
    url = anchor.get_attribute_list('href')[0]
    if url.find("/downloads/rtcoef_rttov12/ir_srf/") >= 0 and url.find(".html") < 0 and url.find("shifted") < 0:
        the_url = "https://nwpsaf.eu/" + url
        get_spectra(the_url)
        
