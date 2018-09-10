
# coding: utf-8

# In[ ]:


import urllib.request
url = "http://uta.pw/shodou/img/28/214.png"
save_name = "test.png"
urllib.request.urlretrieve(url, save_name)
print("saved")


# In[ ]:


import urllib.request
url = "http://uta.pw/shodou/img/28/214.png"
save_name = "test1.png"
#downloding
memory = urllib.request.urlopen(url).read()
with open(save_name, mode="wb") as f:
    f.write(memory)
    print("saved")


# In[ ]:


import urllib.request
url = "http://api.aoikujira.com/ip/ini"
res = urllib.request.urlopen(url)
data = res.read()
text = data.decode("utf-8")
print(text)


# In[ ]:


import urllib.request
import urllib.parse

API = "http://api.aoikujira.com/zip/xml/get.php"
values = {
    "fmt": "xlm", #fmt >> format
    "zn": "5600045"
}
params = urllib.parse.urlencode(values)
url = API + "?" + params
print("url :",url)
data = urllib.request.urlopen(url).read()
text = data.decode("utf-8")
print(data, text)


# In[ ]:


import sys
import urllib.request as request
import urllib.parse as parse
#if you use python shell, use codes blelow
# if len(sys.argv) <= 1:
#     print("usage:" , hyakunin.py(keyword))
#     keyword = sys.argv[1]
#####################################

#if you use jupyter, use codes below
keyword = input()
##########################
print(keyword)
API = "http://api.aoikujira.com/hyakunin/get.php"
query = {
    "fmt": "ini",
    "key": keyword
}
param = parse.urlencode(query)
url = API + "?" + param
print("url: ",url)

with request.urlopen(url) as f:
    bi = f.read()
    data = bi.decode("utf-8")
    print(data)


# In[ ]:


#Let's use Beautiful Soup
from bs4 import BeautifulSoup
html='''
<html>
<body>
<h1>What's is scraping?</h1>
<p>Analize HTML page</p>
<p>Extraction Elements</p>
</body>
</html>'''

soup = BeautifulSoup(html, "html.parser")
print(soup)
h1 = soup.html.body.h1
p1 = soup.html.body.p
p2 = p1.next_sibling.next_sibling

print("h1:", h1.string)
print("p1:", p1.string)
print("p2:", p2.string)


# In[ ]:


from bs4 import BeautifulSoup
html='''
<html>
<body>
<h1 id="title">What's is scraping?</h1>
<p id="txt">Analize HTML page</p>
<p>Extraction Elements</p>
</body>
</html>'''

soup = BeautifulSoup(html, "html.parser")
title = soup.find(id="title")
txt  = soup.find(id="txt")
print("#title: ", title.string)
print("#txt: ", txt.string)


# In[ ]:


from bs4 import BeautifulSoup
html = '''
<html><body>
<ul>
<li><a href="http://uta.pw">uta</a></li>
<li><a href="http://oto.chu.jp">oto</a></li>
</ul>
</body></html>'''

soup = BeautifulSoup(html, "html.parser")
links = soup.find_all("a")
for a in links:
    href = a.attrs["href"]
    text = a.string
    print(text, ">", href)


# In[ ]:


print(soup.prettify())
print(a.attrs)


# In[ ]:


from bs4 import BeautifulSoup
import urllib.request as request

url = "http://api.aoikujira.com/zip/xml/3300075"
res = request.urlopen(url)

soup = BeautifulSoup(res, "html.parser")

ken = soup.find("ken").string
shi = soup.find("shi").string
cho = soup.find("cho").string
print("ken, shi, cho >>", ken, shi, cho)


# In[ ]:


from bs4 import BeautifulSoup
html = '''
<html><body>
<div id="works">
<h1>紅玉いづきの作品リスト</h1>
<ul class="items">
<li>ミミズクと夜の王</li>
<li>MAMA</li>
<li>雪蟷螂</li>
</ul>
</div>
</body></html>'''

soup = BeautifulSoup(html, "html.parser")
h1 = soup.select_one("div#works > h1").string
print("h1: ",h1)
li_list = soup.select("div#works > ul.items > li")
print(li_list)
for li in li_list:
    print("li: ", li.string)


# In[ ]:


from bs4 import BeautifulSoup
import urllib.request as request

url = "http://stocks.finance.yahoo.co.jp/stocks/detail/?code=usdjpy"
response = request.urlopen(url)
soup = BeautifulSoup(response, "html.parser")

price = soup.select_one(".stoksPrice").string
print("usd/jpy: ", price)


# In[ ]:


from bs4 import BeautifulSoup
import urllib.request as request
url = "http://api.aoikujira.com/kawase/xml/usd"
response = request.urlopen(url)
soup = BeautifulSoup(response, "html.parser")
print(soup)
jpy = soup.select_one("jpy").string 
print("usd/jpy: ", jpy)


# In[ ]:


from bs4 import BeautifulSoup
import urllib.request as request

url = "http://www.aozora.gr.jp/index_pages/person148.html"
response = request.urlopen(url)
soup = BeautifulSoup(response, "html.parser")

li_list = soup.select("ol > li")
for li in li_list:
    a = li.a
    if a != None:
        name = a.string
        href = a.attrs["href"]
        print(name, ">", href)


# In[ ]:


from bs4 import BeautifulSoup
# fp = open("books.html", encoding="utf-8")
# soup = BeautifulSoup(fp, "html.parser")
html = '''
<html><body>
<ul id="bible">
<li id="num">Numbers</li>
</ul>
</html></body>'''

soup = BeautifulSoup(html, "html.parser")
sel = lambda q: print(soup.select_one(q).string)
sel("#num")
sel("li#num")
sel("ul > li#num")
sel("#bible > #num")
sel("ul#bible > li#num")
sel("ul#bible > #num")
sel("li[id='num']")
sel("li:nth-of-type(1)")
print(soup.select("li")[0].string)
print(soup.find_all("li")[0].string)


# In[ ]:


from bs4 import BeautifulSoup
fp = open("fruits-vegetables.html", encoding="utf-8")
soup = BeautifulSoup(fp, "html.parser")
print(soup.select_one("li:nth-of-type(8)").string)
print(soup.select_one("#ve-list > li:nth-of-type(4)").string)
print(soup.select("#ve-list > li[data-lo='us']")[1].string)
print(soup.select("#ve-list > li.black")[1].string)
condition = {"data-lo": "us", "class": "black"}
print(soup.find("li", condition).string)
print(soup.find(id="ve-list").find("li", condition).string)


# In[ ]:


from bs4 import BeautifulSoup
import re
html = '''<ul>
<li><a href="first.html">first</li>
<li><a href="https://example.com/second">second</li>
<li><a href="https://example.com/third">third</li>
<li><a href="http://example.com"/forth>forth</li>
</ul>'''
soup = BeautifulSoup(html, "html.parser")
li = soup.find_all(href=re.compile(r"^https://"))
print(li)
for e in li: print(e.attrs["href"])


# In[ ]:


from urllib.parse import urljoin
base = "http://example.com/html/a.html"
print(urljoin(base, "b.html"))
print(urljoin(base, "sub/c.html"))
print(urljoin(base, "../index.html"))
print(urljoin(base, "../img/hoge.html"))
print(urljoin(base, "../css/hoge.css"))


# In[ ]:


from urllib.parse import urljoin

base = "http://example.com/html/a.html"
print(urljoin(base, "/hoge.html"))
print(urljoin(base, "http://kaufen.com/wiki"))
print(urljoin(base, "//ute.pw/shodow"))


# In[ ]:


from bs4 import BeautifulSoup
import urllib.request 
import urllib.parse
import  time, re
import os
proc_files = {}

#Extract Files
def enum_links(html, base):
    soup = BeautifulSoup(html, "html.parser")
    links = soup.select("link[rel='stylesheet']")
    links += soup.select("a[href]")
    result = []
    for a in links:
        href = a.attrs["href"]
        print(href)
        url = urllib.parse.urljoin(base, href)
        result.append(url)
    return result

def download_files(url):
    output = urllib.parse.urlparse(url)
    savepath = "./" + output.netloc + output.path
    if re.search(r"/$", savepath):
        savepath += "index.html"
    savedir = os.path.dirname(savepath)
    
    #download files
    if not os.path.exists(savedir):
        print("mkdir:", savedir)
        os.makedirs(savedir)
    try:
        print("download: ", url)
        urllib.request.urlretrieve(url, savepath)
        time.sleep(1)
        return savepath
    except:
        print("Failure to download: ", url)
        return None

#HTML analyzer and downloader
def analize_html(url, root_url):
    savepath = download_files(url)
    if savepath is None:
        return print("savepath doesn't exist")
    if savepath in proc_files:
        return print("already analized")
    proc_files[savepath] = True
    print("analyze html:", savepath)
    html = open(savepath, "r", encoding="utf-8").read()
    links = enum_links(html, url)
    for link_url in links:
        if link_url.find(root_url) != 0:
            if not re.search(r".css$", link_url):
                continue
            if re.search(r".(html|htm)$", link_url):
                analize_html(link_url, root_url)
            download_files(link_url)
            
url = "http://docs.python.jp/3.6/library/"
analize_html(url, url)

