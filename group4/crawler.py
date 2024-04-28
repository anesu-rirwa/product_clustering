import streamlit as st
import hashlib
from io import BytesIO
from PIL import Image
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver import ChromeOptions


def get_images(url):
  options = ChromeOptions()
  options.add_argument("--headless=new")
  driver = webdriver.Chrome(options=options)
  driver.get(url)
  content = driver.page_source
  soup = BeautifulSoup(content, "html.parser")
  driver.quit()

  results = []
  for a in soup.findAll(attrs={"class": "s-item__image-wrapper image-treatment"}):
    image_url = a.find("img").get("src")
    if image_url not in results:
      results.append(image_url)
  return results


def main():
  st.title("eBay Image Scraper")
  url = st.text_input("Enter eBay Listing URL")
  if st.button("Scrape Images"):
    if url:
      st.write("Fetching images...")
      images = get_images(url)
      if images:
        st.success(f"Found {len(images)} images!")
        for image_url in images:
          try:
            image_content = requests.get(image_url).content
            image_file = BytesIO(image_content)
            image = Image.open(image_file).convert("RGB")
            st.image(image)
          except Exception as e:
            st.error(f"Error downloading image: {e}")
      else:
        st.warning("No images found on the listing.")
    else:
      st.warning("Please enter a valid eBay listing URL.")


if __name__ == "__main__":
  main()
