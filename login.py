import logging
from kiteconnect import KiteConnect
import os
from dotenv import load_dotenv; 


loaded=load_dotenv()
print("load_dotenv returned:", loaded)


logging.basicConfig(level= logging.DEBUG)

API_KEY = os.environ.get('KITE_API_KEY')
api_secret =os.environ.get('KITE_API_SECRET')


if not API_KEY:
    raise ValueError("KITE_API_KEY environment variable must be set.")

kite = KiteConnect(api_key= API_KEY)

kite.login_url()

data = kite.generate_session("request_token_here", api_secret="your_secret")
kite.set_access_token(data["access_token"])


# Optionally Place an order
try:
    order_id = kite.place_order(tradingsymbol="INFY",
                                exchange=kite.EXCHANGE_NSE,
                                transaction_type=kite.TRANSACTION_TYPE_BUY,
                                quantity=1,
                                variety=kite.VARIETY_AMO,
                                order_type=kite.ORDER_TYPE_MARKET,
                                product=kite.PRODUCT_CNC,
                                validity=kite.VALIDITY_DAY)

    logging.info("Order placed. ID is: {}".format(order_id))
except Exception as e:
    logging.info("Order placement failed: {}".format(e.message))