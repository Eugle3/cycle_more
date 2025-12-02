import requests

def test():
    print("This is a test function.")
    

def call_turbo(start_lat, start_lon, end_lat, end_lon):
    url = "https://maps.mail.ru/osm/tools/overpass/api/interpreter"
    query = f"""
        [out:json][timeout:900];
        (
        relation["route"="bicycle"]({start_lat},{start_lon},{end_lat},{end_lon});
        );
        out geom;
        """
    response = requests.post(url, data={"data": query})
    data = response.json()
    print("Turbo API response data:")
    print(data)
    return data

def extract_turbo_route(data):
    route = []
    for route in data['elements']:
        if 'members' in route:
            for member in route['members']:
                if member['type'] == 'way' and 'geometry' in member:
                    for point in member['geometry']:
                        route.append((point['lat'], point['lon']))
        
    return route

# call_turbo(52.5200, 13.4050, 53.5206, 15.4094)
# test()


if __name__ == "__main__":
# Uncomment the function you want to run:
    test()
# call_turbo(52.5200, 13.4050, 53.5206, 15.4094)