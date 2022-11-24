from alpaca_trade_api.rest import REST, TimeFrame
from alpaca_trade_api.stream import Stream

API_KEY = 'PKDT6FRQ9HW6SFH9J90Y'
SECRET_KEY = 'wjTXKSqD8PLzDfdbRrIaYhFrTDryiJMQugctWgoN'
rest_api = REST(API_KEY, SECRET_KEY, 'https://paper-api.alpaca.markets')