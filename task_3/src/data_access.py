from pathlib import Path
import pandas as pd, yaml
import joblib
from sqlalchemy import create_engine


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_pickle(path: Path):
    return joblib.load(path)


def load_feature_list(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

def create_ML_TABLE(connection_string: str)-> pd.DataFrame:
    engine = create_engine(connection_string)
    tables={
    'customers':'customers',
    'geo_location':'geo_location',
    'sellers':'sellers',
    'products':'products',
    'orders':'orders',
    'order_items':'order_items',
    'order_payments':'order_payments',
    'order_reviews':'order_reviews'
    }
    query=f"""select * from {tables['customers']}"""
    customers=pd.read_sql(query,engine)
    query=f"""SELECT * FROM {tables['orders']}"""
    orders=pd.read_sql(query,engine)
    query=f"""SELECT * FROM {tables['order_items']}"""
    order_items=pd.read_sql(query,engine)
    query=f"""SELECT * FROM {tables['order_payments']}"""
    order_payments=pd.read_sql(query,engine)
    query=f"""SELECT * FROM {tables['sellers']}"""
    sellers=pd.read_sql(query,engine)
    query=f"""SELECT * FROM {tables['products']}"""
    products=pd.read_sql(query,engine)
    query=f"""SELECT * FROM {tables['geo_location']}"""
    geo_location=pd.read_sql(query,engine)

    order_items_full = order_items.merge(sellers, on='seller_id', how='left')
    order_items_full = order_items_full.merge(products, on='product_id', how='left')
    order_items_full = order_items_full.merge(
        geo_location.rename(columns={
            'zipcode': 'seller_zipcode',
            'latitude': 'seller_lat',
            'longitude': 'seller_lng'
        }),
        on='seller_zipcode', how='left'
    )
    top_categories = order_items_full['product_category'].value_counts().nlargest(20).index
    order_items_full['product_category_grouped'] = order_items_full['product_category'].where(
        order_items_full['product_category'].isin(top_categories), 'other'
    )

    category_dummies = pd.get_dummies(order_items_full['product_category_grouped'], prefix='cat')
    state_dummies = pd.get_dummies(order_items_full['seller_state'], prefix='sel_state')
    order_items_full = order_items_full.drop(columns=['product_category_grouped', 'seller_state','product_category'])
    order_items_full = pd.concat([order_items_full, category_dummies, state_dummies], axis=1)

    state_cols = [col for col in order_items_full.columns if col.startswith('sel_state_')]
    category_cols = [col for col in order_items_full.columns if col.startswith('cat_')]
    agg_dict = {
        'num_sellers': ('seller_id', 'nunique'),
        'num_items': ('order_item_id', 'count'),

        'total_price': ('price', 'sum'),
        'total_freight_value': ('freight_value', 'sum'),

        'avg_seller_lat': ('seller_lat', 'mean'),
        'avg_seller_lng': ('seller_lng', 'mean'),

        'max_product_weight_grams': ('product_weight_grams', 'max'),
        'max_product_length_cm': ('product_length_cm', 'max'),
        'max_product_height_cm': ('product_height_cm', 'max'),
        'max_product_width_cm': ('product_width_cm', 'max'),
        'max_shipping_limit_date': ('shipping_limit_date', 'max'),

        'seller_zipcode': ('seller_zipcode', 'first'),
        'seller_city': ('seller_city', 'first'),

        'avg_product_name_length': ('product_name_length', 'mean'),
        'avg_product_description_length': ('product_desc_length', 'mean'),
        'avg_product_photos_qty': ('product_photos_qty', 'mean'),
    }
    for col in state_cols + category_cols:
        agg_dict[col] = (col, 'sum')

    order_items_agg = order_items_full.groupby('order_id').agg(**agg_dict).reset_index()
    payment_dummies = pd.get_dummies(order_payments['payment_type'], prefix='pay')
    order_payments_full = pd.concat([order_payments, payment_dummies], axis=1)

    payment_dummy_cols = [c for c in order_payments_full.columns if c.startswith('pay_')]


    agg_dict = {
        'num_payments': ('payment_sequential', 'count'),
        'num_installments': ('payment_installments', 'sum'),
        'total_payment_value': ('payment_value', 'sum'),
    }
    for col in payment_dummy_cols:
        agg_dict[col] = (col, 'sum')

    order_payments_agg = order_payments_full.groupby('order_id').agg(**agg_dict).reset_index()
    customers_and_zipcodes = customers.merge(   
        geo_location.rename(columns={
            'zipcode': 'customer_zipcode',
            'latitude': 'customer_lat',
            'longitude': 'customer_lng'
        }),
        on='customer_zipcode', how='left')

    final_table = orders.merge(order_items_agg, on='order_id', how='left')
    final_table = final_table.merge(order_payments_agg, on='order_id', how='left')
    final_table = final_table.merge(customers_and_zipcodes, on='customer_id', how='left')
    final_table = final_table.drop(columns=['city', 'geostate'])

    #p=yaml.safe_load(open('../config/params.yaml'))
    #final_table.to_csv(p['paths']['ml_table'], index=False)
    return final_table

def create_order_features(order: dict) -> pd.DataFrame:
    connection_string=yaml.safe_load(open('../config/params.yaml'))['connection_string']
    engine = create_engine(connection_string)
    tables={
    'customers':'customers',
    'geo_location':'geo_location',
    'sellers':'sellers',
    'products':'products',
    'orders':'orders',
    'order_items':'order_items',
    'order_payments':'order_payments',
    'order_reviews':'order_reviews'
    }
    query=f"""select * from {tables['customers']}"""
    customers=pd.read_sql(query,engine)
    orders=pd.DataFrame([order])
    query=f"""SELECT * FROM {tables['order_items']}"""
    order_items=pd.read_sql(query,engine)
    query=f"""SELECT * FROM {tables['order_payments']}"""
    order_payments=pd.read_sql(query,engine)
    query=f"""SELECT * FROM {tables['sellers']}"""
    sellers=pd.read_sql(query,engine)
    query=f"""SELECT * FROM {tables['products']}"""
    products=pd.read_sql(query,engine)
    query=f"""SELECT * FROM {tables['geo_location']}"""
    geo_location=pd.read_sql(query,engine)

    order_items_full = order_items.merge(sellers, on='seller_id', how='left')
    order_items_full = order_items_full.merge(products, on='product_id', how='left')
    order_items_full = order_items_full.merge(
        geo_location.rename(columns={
            'zipcode': 'seller_zipcode',
            'latitude': 'seller_lat',
            'longitude': 'seller_lng'
        }),
        on='seller_zipcode', how='left'
    )
    top_categories = order_items_full['product_category'].value_counts().nlargest(20).index
    order_items_full['product_category_grouped'] = order_items_full['product_category'].where(
        order_items_full['product_category'].isin(top_categories), 'other'
    )

    category_dummies = pd.get_dummies(order_items_full['product_category_grouped'], prefix='cat')
    state_dummies = pd.get_dummies(order_items_full['seller_state'], prefix='sel_state')
    order_items_full = order_items_full.drop(columns=['product_category_grouped', 'seller_state','product_category'])
    order_items_full = pd.concat([order_items_full, category_dummies, state_dummies], axis=1)

    state_cols = [col for col in order_items_full.columns if col.startswith('sel_state_')]
    category_cols = [col for col in order_items_full.columns if col.startswith('cat_')]
    agg_dict = {
        'num_sellers': ('seller_id', 'nunique'),
        'num_items': ('order_item_id', 'count'),

        'total_price': ('price', 'sum'),
        'total_freight_value': ('freight_value', 'sum'),

        'avg_seller_lat': ('seller_lat', 'mean'),
        'avg_seller_lng': ('seller_lng', 'mean'),

        'max_product_weight_grams': ('product_weight_grams', 'max'),
        'max_product_length_cm': ('product_length_cm', 'max'),
        'max_product_height_cm': ('product_height_cm', 'max'),
        'max_product_width_cm': ('product_width_cm', 'max'),
        'max_shipping_limit_date': ('shipping_limit_date', 'max'),

        'seller_zipcode': ('seller_zipcode', 'first'),
        'seller_city': ('seller_city', 'first'),

        'avg_product_name_length': ('product_name_length', 'mean'),
        'avg_product_description_length': ('product_desc_length', 'mean'),
        'avg_product_photos_qty': ('product_photos_qty', 'mean'),
    }
    for col in state_cols + category_cols:
        agg_dict[col] = (col, 'sum')

    order_items_agg = order_items_full.groupby('order_id').agg(**agg_dict).reset_index()
    payment_dummies = pd.get_dummies(order_payments['payment_type'], prefix='pay')
    order_payments_full = pd.concat([order_payments, payment_dummies], axis=1)

    payment_dummy_cols = [c for c in order_payments_full.columns if c.startswith('pay_')]


    agg_dict = {
        'num_payments': ('payment_sequential', 'count'),
        'num_installments': ('payment_installments', 'sum'),
        'total_payment_value': ('payment_value', 'sum'),
    }
    for col in payment_dummy_cols:
        agg_dict[col] = (col, 'sum')

    order_payments_agg = order_payments_full.groupby('order_id').agg(**agg_dict).reset_index()
    customers_and_zipcodes = customers.merge(   
        geo_location.rename(columns={
            'zipcode': 'customer_zipcode',
            'latitude': 'customer_lat',
            'longitude': 'customer_lng'
        }),
        on='customer_zipcode', how='left')

    final_row = orders.merge(order_items_agg, on='order_id', how='left')
    final_row = final_row.merge(order_payments_agg, on='order_id', how='left')
    final_row = final_row.merge(customers_and_zipcodes, on='customer_id', how='left')
    final_row = final_row.drop(columns=['city', 'geostate'])

    return final_row