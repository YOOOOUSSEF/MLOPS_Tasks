CREATE TABLE product_translation(
  category varchar(50) UNIQUE,
  category_translation varchar(50) UNIQUE
);

CREATE TABLE geo_location(
  zipcode char(5) PRIMARY KEY,
  latitude double precision,
  longitude double precision,
  city varchar(50),
  geostate char(2)
);

CREATE TABLE customers(
  customer_id varchar(50),
  customer_unique_id varchar(50),
  customer_zipcode char(5) REFERENCES geo_location(zipcode),
  customer_city varchar(50),
  customer_state char(2),
  PRIMARY KEY(customer_id)
);

CREATE TABLE sellers(
  seller_id varchar(50) PRIMARY KEY,
  seller_zipcode char(5) REFERENCES geo_location(zipcode),
  seller_city varchar(50),
  seller_state varchar(5)
);

CREATE TABLE products(
  product_id varchar(50) PRIMARY KEY,
  product_category varchar(50),
  product_name_length smallint,
  product_desc_length smallint,
  product_photos_qty smallint,
  product_weight_grams integer,
  product_length_cm smallint,
  product_height_cm smallint,
  product_width_cm smallint
);

CREATE TABLE orders(
  order_id varchar(50) PRIMARY KEY,
  customer_id varchar(50) REFERENCES customers (customer_id),
  order_status varchar(50),
  order_purchase timestamp,
  order_approved timestamp,
  order_delivered_carrier timestamp,
  order_delivered_customer timestamp,
  order_estimated_delivery timestamp
);

CREATE TABLE order_payments(
  order_id varchar(50) REFERENCES orders (order_id),
  payment_sequential smallint,
  payment_type varchar(20),
  payment_installments smallint,
  payment_value double precision,
  PRIMARY KEY (order_id,payment_sequential)
);

CREATE TABLE order_reviews(
  review_id varchar(50),
  order_id varchar(50) REFERENCES orders (order_id),
  review_score smallint,
  review_title text,
  review_comment text,
  review_create timestamp,
  review_answer timestamp,
   PRIMARY KEY(review_id,order_id)
);

CREATE TABLE order_items(
  order_id varchar(50) REFERENCES orders (order_id),
  order_item_id smallint,
  product_id varchar(50) REFERENCES products (product_id),
  seller_id varchar(50) REFERENCES sellers (seller_id),
  shipping_limit_date timestamp,
  price real,
  freight_value real,
  PRIMARY KEY (order_id,order_item_id)
);
