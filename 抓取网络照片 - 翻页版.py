# -*- coding:utf8 -*-
import requests
import re
from urllib import parse
import os
import time

class BaiduImageSpider(object):
    def __init__(self):
        # 使用 Baidu 图片搜索的 JSON 接口
        self.url = 'https://image.baidu.com/search/acjson?tn=resultjson_com&logid=undefined&ipn=rj&ct=201326592&fp=result&queryWord={}&cl=2&lm=-1&ie=utf-8&oe=utf-8&word={}&pn={}&rn=30'
        self.headers = {'User-Agent':'Mozilla/5.0'}

    # 获取单个页面的图片
    def get_image(self, url, word, total_downloaded):
        res = requests.get(url, headers=self.headers)
        res.encoding = "utf-8"
        try:
            json_data = res.json()
        except ValueError as e:
            print(f"解析 JSON 失败，原因：{e}")
            return 0, True  # 返回 True 表示没有更多图片

        # 从 JSON 数据中提取图片链接
        data_list = json_data.get('data', [])
        # 移除空的字典
        data_list = [item for item in data_list if item and 'hoverURL' in item]

        if not data_list:
            return 0, True  # 没有更多图片

        img_link_list = [item.get('hoverURL') for item in data_list if item.get('hoverURL')]

        if not img_link_list:
            return 0, False  # 这一页没有图片，但可能还有下一页

        directory = 'D:/image/{}/'.format(word)
        if not os.path.exists(directory):
            os.makedirs(directory)

        num_successful_downloads = 0
        for img_link in img_link_list:
            if total_downloaded + num_successful_downloads >= 150:
                break
            filename = '{}{}_{}.jpg'.format(directory, word, total_downloaded + num_successful_downloads + 1)
            success = self.save_image(img_link, filename)
            if success:
                num_successful_downloads += 1
        return num_successful_downloads, False  # False 表示可能还有更多图片

    # 下载单张图片
    def save_image(self, img_link, filename):
        if not img_link or not (img_link.startswith('http://') or img_link.startswith('https://')):
            print(f'无效的 URL：{img_link}')
            return False

        try:
            response = requests.get(url=img_link, headers=self.headers, timeout=10)
            if response.status_code == 200:
                with open(filename, 'wb') as f:
                    f.write(response.content)
                print(filename, '下载成功')
                return True
            else:
                print(f'下载失败 {img_link}。HTTP 状态码：{response.status_code}')
                return False
        except Exception as e:
            print(f"下载失败 {img_link}。原因：{e}")
            return False

    # 处理分页和控制下载总数
    def get_images(self, word, word_parse):
        total_downloaded = 0
        pn = 0  # 分页参数

        while total_downloaded < 150:
            url = self.url.format(word_parse, word_parse, pn)
            num_successful_downloads, no_more_images = self.get_image(url, word, total_downloaded)
            if no_more_images:
                print('没有更多图片可供下载。')
                break
            total_downloaded += num_successful_downloads
            pn += 30  # 每页30张图片，递增分页参数
            time.sleep(1)  # 礼貌地等待一秒再发送下一次请求
        print('共下载了 {} 张图片。'.format(total_downloaded))

    # 程序入口
    def run(self):
        word = input("您想要谁的照片？")
        word_parse = parse.quote(word)
        self.get_images(word, word_parse)

if __name__ == '__main__':
    spider = BaiduImageSpider()
    spider.run()
