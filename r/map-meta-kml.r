# Import Data and Select Farms Growing Crops

library(tidyverse)
library(sf)

farms_nl <- read_sf('~/Downloads/FARMS.kml')
st_geometry(farms_nl) <- NULL

glimpse(farms_nl)

farm_contact <- farms_nl %>%
  select(Name, Contact, Website, Address, Town, Postal_Code, Email, Phone_1, Phone_2,
         Availability, Latitude, Longitude)


nl_farms_long <- farms_nl %>%
  separate(col = 'Products', sep = ',', into = paste('col', 1:16)) %>%
  select(Name, contains('col')) %>%
  gather(key = 'col', value = 'Product', -Name, na.rm = TRUE) %>%
  select(-col) %>%
  mutate(Product = trimws(Product)) %>%
  mutate(Category = case_when(Product %in% c('Honey/Honey Products', 'Honey',
                                             'Pollination Services', 'Adopt-a-Hive',
                                             'Fireweed Comb Honey', 'Raw Wildflower Honey',
                                             'Honeybees', 'Bees wax', 'Beeswax', 'Honey Bee Rescue',
                                             'Swarm Prevention Program', 'Artisan Fireweed Honey',
                                             'Liquid Honey', 'Comb Honey') ~ 'Bees',
                              Product %in% c('Nursery Sod', 'Turf Sod', 'Sod',
                                             'Turf Grass') ~ 'Sod',
                              Product %in% c('Field Vegetables') ~ 'Field Crops',
                              Product %in% c('Greenhouse Vegetables', 'Winter Greens') ~ 'Greenhouse',
                              Product %in% c('Perennials', 'Hanging Baskets',
                                             'Bedding Plants', 'Flowers', 'Baskets',
                                             'Ornamental Annuals and Perennials',
                                             'Planters', 'Floral Arrangments') ~ 'Flowers',
                              Product %in% c('Berries', 'Cranberries', 'Blueberries',
                                             'Apples', 'Flash Frozen Blueberries', 'U-Pick',
                                             'Fruit Trees', 'Berry Plants') ~ 'Fruit',
                              Product %in% c('Trees', 'Christmas Trees', 'Landscape Trees') ~ 'Trees',
                              Product %in% c('Turkey', 'Lamb', 'Milk/Dairy Products',
                                             'Beef', 'Eggs', 'Wool', 'Sheep and Alpaca Fibre',
                                             'Handspun Yarn', 'Pork', 'Chicken', 'Sausage',
                                             'Llamas', 'Hereford cattle stock for resale',
                                             'Goose', 'Patties', 'Angora Goats', 'Grass Fed Beef',
                                             'Live Chickens', 'Live Ducks', 'Petting Farm',
                                             'Petting Barn', 'Tilapia') ~ 'Animals',
                              Product %in% c('Hay') ~ 'Hay',
                              Product %in% c('Value Added( preserves', 'Tomato Products',
                                             'Bodly care products from beehive', 'Pickles',
                                             'jams', 'Knitted and Handwoven Sweaters',
                                             'Frozen French Fries', 'Beeswax candles',
                                             'Jams', 'Value-Added', 'Honey Cranberry Juice',
                                             'Shawl', 'Primal cuts & Secondary products(sausages',
                                             'Preserves', 'Pre-Peeled Potatoes', 'Jellies',
                                             'Honey Berry Sauces', 'ground and meatballs)',
                                             'Baked Goods', 'Knitted Goods', 'Homemade Crafts',
                                             'Frozen Products', 'Infused Oils', 'Balms',
                                             'Fermented Products', 'Salves', 'Bottled Items',
                                             'Bakery') ~ 'Value Added',
                              Product %in% c('Herbs') ~ 'Herbs',
                              Product %in% c('Microgreens', 'Shoots and Micro-Greens') ~ 'Microgreens',
                              TRUE ~ 'Other'))

# Adding Contact Infor for Farms from NLGROWN Google Map
# Source: https://www.google.com/maps/d/u/0/viewer?mid=1GgAYy8lc8Gj-IogE62MGxAmnZKWOqYF-&hl=en&ll=47.470303%2C-52.78436899999999&z=8
df_farms <- nl_farms_long %>%
  filter(Category %in% c('Bees', 'Field Crops', 'Greenhouse', 'Fruit', 'Herbs', 'Microgreens')) %>%
  group_by(Name) %>%
  mutate(Products = paste(Product, collapse = ', '),  Categories = paste(Category, collapse = ', ')) %>%
  left_join(farm_contact, by = 'Name') %>%
  distinct() %>%
  select(-Product, -Category)

write_tsv(df_farms, file = '~/work/Data Farms/df_farms_contact.tsv')

getwd()