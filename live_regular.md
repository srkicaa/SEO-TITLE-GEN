---
title: "Live Google Organic SERP Regular"
url: "https://docs.dataforseo.com/v3/serp/google/organic/live/regular/"
date: "2025-04-25"
---

## Live Google Organic SERP Regular

  
Live SERP provides real-time data on top 100 search engine results for the specified keyword, search engine, and location.

 

 

> Instead of ‘login’ and ‘password’ use your credentials from https://app.dataforseo.com/api-access

```
# Instead of 'login' and 'password' use your credentials from https://app.dataforseo.com/api-access \
login="login" 
password="password" 
cred="$(printf ${login}:${password} | base64)" 
curl --location --request POST "https://api.dataforseo.com/v3/serp/google/organic/live/regular" \
--header "Authorization: Basic ${cred}"  \
--header "Content-Type: application/json" \
--data-raw '[
    {
        "language_code": "en",
        "location_code": 2840,
        "keyword": "albert einstein"
    }
]'

```

```php
<?php
// You can download this file from here https://cdn.dataforseo.com/v3/examples/php/php_RestClient.zip
require('RestClient.php');
$api_url = 'https://api.dataforseo.com/';
try {
	// Instead of 'login' and 'password' use your credentials from https://app.dataforseo.com/api-access
	$client = new RestClient($api_url, null, 'login', 'password');
} catch (RestClientException $e) {
	echo "\n";
	print "HTTP code: {$e->getHttpCode()}\n";
	print "Error code: {$e->getCode()}\n";
	print "Message: {$e->getMessage()}\n";
	print  $e->getTraceAsString();
	echo "\n";
	exit();
}
$post_array = array();
// You can set only one task at a time
$post_array[] = array(
	"language_code" => "en",
	"location_code" => 2840,
	"keyword" => mb_convert_encoding("albert einstein", "UTF-8")
);
try {
	// POST /v3/serp/google/organic/live/regular
	// in addition to 'google' and 'organic' you can also set other search engine and type parameters
	// the full list of possible parameters is available in documentation
	$result = $client->post('/v3/serp/google/organic/live/regular', $post_array);
	print_r($result);
	// do something with post result
} catch (RestClientException $e) {
	echo "\n";
	print "HTTP code: {$e->getHttpCode()}\n";
	print "Error code: {$e->getCode()}\n";
	print "Message: {$e->getMessage()}\n";
	print  $e->getTraceAsString();
	echo "\n";
}
$client = null;
?>
```

```python
from client import RestClient
# You can download this file from here https://cdn.dataforseo.com/v3/examples/python/python_Client.zip
client = RestClient("login", "password")
post_data = dict()
# You can set only one task at a time
post_data[len(post_data)] = dict(
    language_code="en",
    location_code=2840,
    keyword="albert einstein"
)
# POST /v3/serp/google/organic/live/regular
# in addition to 'google' and 'organic' you can also set other search engine and type parameters
# the full list of possible parameters is available in documentation
response = client.post("/v3/serp/google/organic/live/regular", post_data)
# you can find the full list of the response codes here https://docs.dataforseo.com/v3/appendix/errors
if response["status_code"] == 20000:
    print(response)
    # do something with result
else:
    print("error. Code: %d Message: %s" % (response["status_code"], response["status_message"]))


```

```js
const axios = require('axios');

axios({
    method: 'post',
    url: 'https://api.dataforseo.com/v3/serp/google/organic/live/regular',
    auth: {
        username: 'login',
        password: 'password'
    },
    data: [{
        "keyword": encodeURI("albert einstein"),
        "language_code": "en",
        "location_code": 2840
    }],
    headers: {
        'content-type': 'application/json'
    }
}).then(function (response) {
    var result = response['data']['tasks'];
    // Result data
    console.log(result);
}).catch(function (error) {
    console.log(error);
});

```

```csharp
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Threading.Tasks;

namespace DataForSeoDemos
{
    public static partial class Demos
    {
        public static async Task serp_live_regular()
        {
            var httpClient = new HttpClient
            {
                BaseAddress = new Uri("https://api.dataforseo.com/"),
                // Instead of 'login' and 'password' use your credentials from https://app.dataforseo.com/api-access
                //DefaultRequestHeaders = { Authorization = new AuthenticationHeaderValue("Basic", Convert.ToBase64String(Encoding.ASCII.GetBytes("login:password"))) }
            };
            var postData = new List<object>();
            // You can set only one task at a time
            postData.Add(new
            {
                language_code = "en",
                location_code = 2840,
                keyword = "albert einstein"
            });
            // POST /v3/serp/google/organic/live/regular
            // in addition to 'google' and 'organic' you can also set other search engine and type parameters
            // the full list of possible parameters is available in documentation
            var taskPostResponse = await httpClient.PostAsync("/v3/serp/google/organic/live/regular", new StringContent(JsonConvert.SerializeObject(postData)));
            var result = JsonConvert.DeserializeObject<dynamic>(await taskPostResponse.Content.ReadAsStringAsync());
            // you can find the full list of the response codes here https://docs.dataforseo.com/v3/appendix/errors
            if (result.status_code == 20000)
            {
                // do something with result
                Console.WriteLine(result);
            }
            else
                Console.WriteLine($"error. Code: {result.status_code} Message: {result.status_message}");
        }
    }
}

```

> The above command returns JSON structured like this:

```
{
  "version": "0.1.20200129",
  "status_code": 20000,
  "status_message": "Ok.",
  "time": "12.2397 sec.",
  "cost": 0.003,
  "tasks_count": 1,
  "tasks_error": 0,
  "tasks": [
    {
      "id": "01301234-1535-0121-0000-9c8e396a59cc",
      "status_code": 20000,
      "status_message": "Ok.",
      "time": "12.1329 sec.",
      "cost": 0.003,
      "result_count": 1,
      "path": [
        "v3",
        "serp",
        "google",
        "organic",
        "live",
        "regular"
      ],
      "data": {
        "api": "serp",
        "function": "live",
        "se": "google",
        "se_type": "organic",
        "language_name": "English",
        "location_name": "United States",
        "keyword": "flight ticket new york san francisco",
        "tag": "tag2",
        "device": "desktop",
        "os": "windows"
      },
      "result": [
        {
          "keyword": "flight ticket new york san francisco",
          "type": "organic",
          "se_domain": "google.com",
          "location_code": 2840,
          "language_code": "en",
          "check_url": "https://www.google.com/search?q=flight%20ticket%20new%20york%20san%20francisco&num=100&hl=en&gl=US&gws_rd=cr&ie=UTF-8&oe=UTF-8&uule=w+CAIQIFISCQs2MuSEtepUEUK33kOSuTsc",
          "datetime": "2019-11-15 12:57:46 +00:00",
          "spell": null,
          "refinement_chips": {
            "type": "refinement_chips",
            "xpath": "/html[1]/body[1]/div[3]/div[1]/div[9]/div[1]/div[1]/div[3]/div[1]/div[1]",
            "items": [
              {
                "type": "refinement_chips_element",
                "title": "Remote",
                "url": "https://www.google.com/search?num=100&sca_esv=427163c40a0d98b7&hl=en&gl=US&glp=1&q=bristol+temp+agencies+remote&uds=ADvngMjcH0KdF7qGWtwTBrP0nt7drlQNXJ_q6WhUnfcnlFQAOVqvQ09aiEx7jUP4Wb5sg6FzKTGyEoBQg36hfgreicnnhtBQv8H25iRkUIMWBSqTcu0nGLObB57jKDn4sIHqgjkv6AqtXzA0gTV6n6-p1_aOUPMmYdgJOPy5xybgxI0ZY3-OZUg&sa=X&ved=2ahUKEwjEjMPzrvyIAxVgpIkEHYTyBqEQxKsJegQIHRAB&ictx=0",
                "domain": "www.google.com",
                "options": null
              },
              {
                "type": "refinement_chips_element",
                "title": "Date posted",
                "url": null,
                "domain": null,
                "options": [
                  {
                    "type": "refinement_chips_option",
                    "title": "Yesterday",
                    "url": "https://www.google.com/search?num=100&sca_esv=427163c40a0d98b7&hl=en&gl=US&glp=1&q=bristol+temp+agencies+since+yesterday&uds=ADvngMjcH0KdF7qGWtwTBrP0nt7d1cxhUU_4I1tnZ_YIEcACz8ZYvzwewv2vtaifFRGAtrClbFOcrHbTAbAeXm1jREcC6VS1VsCvY-sITnX4ozb-ILgfWEVwq_Z5ROUTUSIPShEnKXD5sUUbZbocrG609xSmt2d98g8y8m1lGjX2kp2G7tWTnMoyaYhx7tPHFsc1SlxiTTuylOmQpyaET98nEnMhDFUw8hSQnpfQcpEDEdBLwrOxN3gOZC4RtcHuKTyeCrFRnaDjQ17dNosh1yiBWrSXB9e1dQF-1Pt17wXCHwRUCH6wH0DbAN-oCNd7lMT2v24eI8rDkLQP3kTLpzGMUZReaWIAPw&sa=X&ved=2ahUKEwjEjMPzrvyIAxVgpIkEHYTyBqEQkbEKegQIDxAD",
                    "domain": "www.google.com"
                  },
                  {
                    "type": "refinement_chips_option",
                    "title": "Last 3 days",
                    "url": "https://www.google.com/search?num=100&sca_esv=427163c40a0d98b7&hl=en&gl=US&glp=1&q=bristol+temp+agencies+in+the+last+3+days&uds=ADvngMjcH0KdF7qGWtwTBrP0nt7dLQVdkfQJDu14-gF_eQoKa39tQdXyMUfu3HnnM3Pu4g5hOzwEez_H8t-yULnPJEgal7YGml0mtSaIuH3YeuIhPq5mBSo_ECo7hOJVwSRGkksXQg-RdTy8GhoT8Qm-ZFkyAQnSLmnlmv0Rm042YAlQ6JFqnfOW0VIOgTeTdYm6SLpzQ3a6YSvEf2tau3pEKxnGWKleL613vj0tLBWg4Jhsa0trdMkFPHDQYhc2caGQ8cmSYU4cqdB2Eq3PsNGgjgebvt7_7bbvjixC19O4L9ZB02rwY0EP30_mYAeTG9lYAx8ZJvGDslVLp0DHS93P8ynaiA7Bbw&sa=X&ved=2ahUKEwjEjMPzrvyIAxVgpIkEHYTyBqEQkbEKegQIDxAE",
                    "domain": "www.google.com"
                  }
                ]
              }
            ]
          },
          "item_types": [
            "paid",
            "organic",
            "ai_overview"
          ],
          "se_results_count": 85600000,
          "items_count": 96,
          "items": [
            {
              "type": "featured_snippet",
              "rank_group": 1,
              "rank_absolute": 1,
              "domain": "www.rome.net",
              "title": "Rome Metro - Lines, hours, fares and Rome metro maps",
              "description": "Most important metro stationsCipro - Musei Vaticani: The closest stop to the Vatican Museums and to the Sistine Chapel.Ottaviano - San Pietro: This station is a few minutes' walk from St. Peter's Square and St. Peter's Basilica.Spagna: Very convenient for visiting Piazza di Spagna and Villa Borghese.More items...",
              "url": "https://www.rome.net/metro",
              "breadcrumb": null
            },
            {
              "type": "paid",
              "rank_group": 1,
              "rank_absolute": 2,
              "domain": "www.bookingbuddy.com",
              "title": "Flights To Lwo | Unbelievably Cheap Flights | BookingBuddy.com‎",
              "description": "Compare Airlines & Sites. Cheap Flights on BookingBuddy, a TripAdvisor Company",
              "url": "https://www.bookingbuddy.com/en/hero/",
              "breadcrumb": "www.bookingbuddy.com/Flights"
            },
            {
              "type": "paid",
              "rank_group": 2,
              "rank_absolute": 3,
              "domain": "www.trip.com",
              "title": "Cheap Flight Tickets | Search & Find Deals on Flights | trip.com‎",
              "description": "Wide Selection of Cheap Flights Online. Explore & Save with Trip.com! Fast, Easy & Secure...",
              "url": "https://www.trip.com/flights/index?utm_campaign=GG_SE_All_en_Flight_Generic_NA_Phrase",
              "breadcrumb": "www.trip.com/"
            },
            {
              "type": "paid",
              "rank_group": 3,
              "rank_absolute": 4,
              "domain": "www.kayak.com",
              "title": "Find the Cheapest Flights | Search, Compare & Save Today‎",
              "description": "Cheap Flights, Airline Tickets and Flight Deals. Compare 100s of Airlines Worldwide. Search...",
              "url": "https://www.kayak.com/horizon/sem/flights/general",
              "breadcrumb": "www.kayak.com/flights"
            },
            {
              "type": "organic",
              "rank_group": 1,
              "rank_absolute": 5,
              "domain": "www.kayak.com",
              "title": "Cheap Flights from New York to San Francisco from $182 ...",
              "description": "Fly from New York to San Francisco on Frontier from $182, United Airlines from ... the cheapest round-trip tickets were found on Frontier ($182), United Airlines ...",
              "url": "https://www.kayak.com/flight-routes/New-York-NYC/San-Francisco-SFO",
              "breadcrumb": "https://www.kayak.com › Flights › North America › United States › California"
            },
            {
              "type": "organic",
              "rank_group": 2,
              "rank_absolute": 6,
              "domain": "www.skyscanner.com",
              "title": "Cheap flights from New York to San Francisco SFO from $123 ...",
              "description": "Flight information New York to San Francisco International .... tool will help you find the cheapest tickets from New York in San Francisco in just a few clicks.",
              "url": "https://www.skyscanner.com/routes/nyca/sfo/new-york-to-san-francisco-international.html",
              "breadcrumb": "https://www.skyscanner.com › United States › New York"
            },
            {
              "type": "organic",
              "rank_group": 3,
              "rank_absolute": 7,
              "domain": "www.expedia.com",
              "title": "JFK to SFO: Flights from New York to San Francisco for 2019 ...",
              "description": "Book your New York (JFK) to San Francisco (SFO) flight with our Best Price ... How much is a plane ticket to San Francisco (SFO) from New York (JFK)?.",
              "url": "https://www.expedia.com/lp/flights/jfk/sfo/new-york-to-san-francisco",
              "breadcrumb": "https://www.expedia.com › flights › jfk › sfo › new-york-to-san-francisco"
            },
            {
              "type": "organic",
              "rank_group": 94,
              "rank_absolute": 97,
              "domain": "www.ethiopianairlines.com",
              "title": "Ethiopian Airlines | Book your next flight online and Fly Ethiopian",
              "description": "Fly to your Favorite International Destination with Ethiopian Airlines. Book your Flights Online for Best Offers/Discounts and Enjoy African Flavored Hospitality.",
              "url": "https://www.ethiopianairlines.com/",
              "breadcrumb": "https://www.ethiopianairlines.com"
            },
            {
              "type": "organic",
              "rank_group": 95,
              "rank_absolute": 98,
              "domain": "www.vietnamairlines.com",
              "title": "Vietnam Airlines | Reach Further | Official website",
              "description": "Great value fares with Vietnam Airlines. Book today and save! Skytrax – 4 Star airline. Official website. Earn frequent flyer miles with Lotusmiles.",
              "url": "https://www.vietnamairlines.com/",
              "breadcrumb": "https://www.vietnamairlines.com"
            },
            {
              "type": "organic",
              "rank_group": 96,
              "rank_absolute": 99,
              "domain": "books.google.com",
              "title": "Code of Federal Regulations: 1985-1999",
              "description": "A purchases in New York a round-trip ticket for transportation by air from New York to ... B purchases a ticket in San Francisco for Combination rail and water ...",
              "url": "https://books.google.com/books?id=av3umFsqbAEC&pg=PA305&lpg=PA305&dq=flight+ticket+new+york+san+francisco&source=bl&ots=fJJY5RUS9l&sig=ACfU3U16ejUqNf23jHD32QNCxDCa05Vn9g&hl=en&ppis=_e&sa=X&ved=2ahUKEwjs_4OnouzlAhXJ4zgGHeBcD3oQ6AEwdXoECHEQAQ",
              "breadcrumb": "https://books.google.com › books"
            }
          ]
        }
      ]
    }
  ]
}
```



**`POST https://api.dataforseo.com/v3/serp/google/organic/live/regular`**



Your account will be charged for each request.  
The cost can be calculated on the [Pricing](https://dataforseo.com/pricing/serp/google-organic-serp-api "Pricing") page.

All POST data should be sent in the [JSON](https://en.wikipedia.org/wiki/JSON) format (UTF-8 encoding). The task setting is done using the POST method. When setting a task, you should send all task parameters in the task array of the generic POST array. You can send up to 2000 API calls per minute, each Live SERP API call can contain only one task.

Below you will find a detailed description of the fields you can use for setting a task.

**Description of the fields for setting a task:**

| Field name | Type | Description |
|---|---|---|
| `url` | string | *direct URL of the search query*   optional field   you can specify a direct URL and we will sort it out to the necessary fields. Note that this method is the most difficult for our API to process and also requires you to specify the exact language and location in the URL. In most cases, we wouldn’t recommend using this method.   example:   `https://www.google.co.uk/search?q=%20rank%20tracker%20api&hl=en&gl=GB&uule=w+CAIQIFISCXXeIa8LoNhHEZkq1d1aOpZS` |
| `keyword` | string | *keyword*   **required field**   you can specify **up to 700 characters** in the `keyword` field   all %## will be decoded (plus character ‘+’ will be decoded to a space character)   if you need to use the “%” character for your `keyword`, please specify it as “%25”;   if you need to use the “+” character for your `keyword`, please specify it as “%2B”;   if this field contains such parameters as *‘allinanchor:’, ‘allintext:’, ‘allintitle:’, ‘allinurl:’, ‘define:’, ‘filetype:’, ‘id:’, ‘inanchor:’, ‘info:’, ‘intext:’, ‘intitle:’, ‘inurl:’, ‘link:’, ‘site:’*, **the charge per task will be multiplied by 5**   **Note:** queries containing the ‘cache:’ parameter are not supported and will return a validation error |
| `location_name` | string | *full name of search engine location*   **required field if you don’t specify** `location_code` or `location_coordinate`   **if you use this field, you don’t need to specify `location_code` or `location_coordinate`**   you can receive the list of available locations of the search engine with their `location_name` by making a separate request to the `https://api.dataforseo.com/v3/serp/google/locations`   example:   `London,England,United Kingdom` |
| `location_code` | integer | *search engine location code*   **required field if you don’t specify** `location_name` or `location_coordinate`   **if you use this field, you don’t need to specify `location_name` or `location_coordinate`**   you can receive the list of available locations of the search engines with their `location_code` by making a separate request to the `https://api.dataforseo.com/v3/serp/google/locations`   example:   `2840` |
| `location_coordinate` | string | *GPS coordinates of a location*   **required field if you don’t specify** `location_name` or `location_code`   **if you use this field, you don’t need to specify `location_name` or `location_code`**   `location_coordinate` parameter should be specified in the *“latitude,longitude,radius”* format   the maximum number of decimal digits for *“latitude”* and *“longitude”*: 7   the minimum value for *“radius”*: 199.9 (mm)   the maximum value for *“radius”*: 199999 (mm)   example:   `53.476225,-2.243572,200` |
| `language_name` | string | *full name of search engine language*   **required field if you don’t specify** `language_code`   **if you use this field, you don’t need to specify `language_code`**   you can receive the list of available languages of the search engine with their `language_name` by making a separate request to the `https://api.dataforseo.com/v3/serp/google/languages`   example:   `English` |
| `language_code` | string | *search engine language code*   **required field if you don’t specify** `language_name`   **if you use this field, you don’t need to specify `language_name`**   you can receive the list of available languages of the search engine with their `language_code` by making a separate request to the `https://api.dataforseo.com/v3/serp/google/languages`   example:   `en` |
| `device` | string | *device type*   optional field   can take the values:`desktop`, `mobile`   default value: `desktop` |
| `os` | string | *device operating system*   optional field   if you specify `desktop` in the `device` field, choose from the following values: `windows`, `macos`   default value: `windows`   if you specify `mobile` in the `device` field, choose from the following values: `android`, `ios`   default value: `android` |
| `se_domain` | string | *search engine domain*   optional field   we choose the relevant search engine domain automatically according to the location and language you specify   however, you can set a custom search engine domain in this field   example:   `google.co.uk`, `google.com.au`, `google.de`, etc. |
| `depth` | integer | *parsing depth*   optional field   number of results in SERP   default value: `100`   max value: `700`   **Note:** your account will be billed per each SERP containing up to 100 results;   thus, setting a depth above `100` may result in additional charges if the search engine returns more than 100 results;   if the specified depth is higher than the number of results in the response, the difference will be refunded automatically to your account balance |
| `target` | string | *target domain, subdomain, or webpage to get results for*   optional field   a domain or a subdomain should be specified without `https://` and `www.`   note that the results of `target`-specific tasks will only include SERP elements that contain a `url` string;   you can also use a wildcard (‘\*’) character to specify the search pattern in SERP and narrow down the results;   examples:   **`example.com`** – returns results for the website’s home page with URLs, such as `https://example.com`, or `https://www.example.com/`, or `https://example.com/`;   **`example.com*`** – returns results for the domain, including all its pages;   **`*example.com*`** – returns results for the entire domain, including all its pages and subdomains;   **`*example.com`** – returns results for the home page regardless of the subdomain, such as `https://en.example.com`;   **`example.com/example-page`** – returns results for the exact URL;   **`example.com/example-page*`** – returns results for all domain’s URLs that start with the specified string |
| `group_organic_results` | boolean | *display related results*   optional field   if set to `true`, the `related_result` element in the response will be provided as a snippet of its parent organic result;   if set to `false`, the `related_result` element will be provided as a separate organic result;   default value: `true` |
| `max_crawl_pages` | integer | *page crawl limit*   optional field   number of search results pages to crawl   max value: `100`   **Note:** the `max_crawl_pages` and `depth` parameters complement each other;   learn more at [our help center](https://dataforseo.com/help-center/what-is-max-crawl-pages-and-how-does-it-work) |
| `search_param` | string | *additional parameters of the search query*   optional field   [get the list of available parameters and additional details here](https://dataforseo.com/what-are-google-search-parameters-and-how-to-use-them-with-serp-api.html) |
| `tag` | string | *user-defined task identifier*   optional field   *the character limit is 255*   you can use this parameter to identify the task and match it with the result   you will find the specified `tag` value in the `data` object of the response |

  
As a response of the API server, you will receive [JSON](https://en.wikipedia.org/wiki/JSON)-encoded data containing a `tasks` array with the information specific to the set tasks.

**Description of the fields in the results array:**

| Field name | Type | Description |
|---|---|---|
| `version` | string | *the current version of the API* |
| `status_code` | integer | *general status code*   you can find the full list of the response codes [here](https://docs.dataforseo.com/v3/appendix/errors.md)   **Note:** we strongly recommend designing a necessary system for handling related exceptional or error conditions |
| `status_message` | string | *general informational message*   you can find the full list of general informational messages [here](https://docs.dataforseo.com/v3/appendix/errors.md) |
| `time` | string | *execution time, seconds* |
| `cost` | float | *total tasks cost, USD* |
| `tasks_count` | integer | *the number of tasks in the **`tasks`**array* |
| `tasks_error` | integer | *the number of tasks in the **`tasks`** array returned with an error* |
| **`tasks`** | array | *array of tasks* |
| `id` | string | *unique task identifier in our system*   in the [UUID](https://en.wikipedia.org/wiki/Universally_unique_identifier) format |
| `status_code` | integer | *status code of the task*   generated by DataForSEO; can be within the following range: 10000-60000   you can find the full list of the response codes [here](https://docs.dataforseo.com/v3/appendix/errors.md) |
| `status_message` | string | *informational message of the task*   you can find the full list of general informational messages [here](https://docs.dataforseo.com/v3/appendix/errors.md) |
| `time` | string | *execution time, seconds* |
| `cost` | float | *cost of the task, USD* |
| `result_count` | integer | *number of elements in the **`result`** array* |
| `path` | array | *URL path* |
| `data` | object | *contains the same parameters that you specified in the POST request* |
| **`result`** | array | *array of results* |
| `keyword` | string | *keyword received in a POST array*   **keyword is returned with decoded %## (plus character ‘+’ will be decoded to a space character)** |
| `type` | string | *search engine type in a POST array* |
| `se_domain` | string | *search engine domain in a POST array* |
| `location_code` | integer | *location code in a POST array* |
| `language_code` | string | *language code in a POST array* |
| `check_url` | string | *direct URL to search engine results* you can use it to make sure that we provided exact results |
| `datetime` | string | *date and time when the result was received*   in the UTC format: “yyyy-mm-dd hh-mm-ss +00:00”   example:   `2019-11-15 12:57:46 +00:00` |
| `spell` | object | *autocorrection of the search engine*   if the search engine provided results for a keyword that was corrected, we will specify the keyword corrected by the search engine and the type of autocorrection |
| `keyword` | string | *keyword obtained as a result of search engine autocorrection*   the results will be provided for the corrected keyword |
| `type` | string | *type of autocorrection*   possible values:   `did_you_mean`, `showing_results_for`, `no_results_found_for`, `including_results_for` |
| `refinement_chips` | object | *search refinement chips* |
| `type` | string | *type of element = **‘refinement\_chips’*** |
| `xpath` | string | *the [XPath](https://en.wikipedia.org/wiki/XPath) of the element* |
| `items` | array | *items of the element* |
| `type` | string | *type of element = **‘refinement\_chips\_element’*** |
| `title` | string | *title of the element* |
| `url` | string | *search URL with refinement parameters* |
| `domain` | string | *domain in SERP* |
| `options` | array | *further search refinement options* |
| `type` | string | *type of element = **‘refinement\_chips\_option’*** |
| `title` | string | *title of the element* |
| `url` | string | *search URL with refinement parameters* |
| `domain` | string | *domain in SERP* |
| `item_types` | array | *types of search results found in SERP*   contains types of all search results (`items`) found in the returned SERP   possible item types:   `answer_box`, `app`, `carousel`, `multi_carousel`, `featured_snippet`, `google_flights`, `google_reviews`, `images`, `jobs`, `knowledge_graph`, `local_pack`, `map`, `organic`, `paid`, `people_also_ask`, `related_searches`, `people_also_search`, `shopping`, `top_stories`, `twitter`, `video`, `events`, `mention_carousel`, `ai_overview`**note** that this array contains all types of search results found in the returned SERP;   however, this endpoint provides data for `featured_snippet`, `organic` and `paid` types only  to get all items (inlcuding SERP features and rich snippets) found in the returned SERP, please refer to the [Google Organiс Advanced SERP](https://docs.dataforseo.com/v3/serp/google/organic/live/advanced.md) endpoint |
| `se_results_count` | integer | *total number of results in SERP* |
| `items_count` | integer | *the number of results returned in the **`items`** array* |
| **`items`** | array | *items in SERP* |
| **‘featured\_snippet’ element in SERP** |  |  |
| `type` | string | *type of element = **‘featured\_snippet’*** |
| `rank_group` | integer | *group rank in SERP*   position within a group of elements with identical `type` values   positions of elements with different `type` values are omitted from `rank_group` |
| `rank_absolute` | integer | *absolute rank in SERP*   absolute position among all the elements found in SERP **note** values are returned in the ascending order, with values corresponding to advanced SERP features omitted from the results;   to get all items (including SERP features and rich snippets) with their positions, please refer to the [Google Organiс Advanced SERP](https://docs.dataforseo.com/v3/serp/google/organic/live/advanced.md) endpoint |
| `domain` | string | *domain in SERP* |
| `title` | string | *title of the result in SERP* |
| `description` | string | *description of the results element in SERP* |
| `url` | string | *relevant URL* |
| `breadcrumb` | string | always equals `null` |
| **‘organic’ element in SERP** |  |  |
| `type` | string | *type of element = **‘organic’*** |
| `rank_group` | integer | *group rank in SERP*   position within a group of elements with identical `type` values   positions of elements with different `type` values are omitted from `rank_group` |
| `rank_absolute` | integer | *absolute rank in SERP*   absolute position among all the elements found in SERP **note** values are returned in the ascending order, with values corresponding to advanced SERP features omitted from the results  to get all items (including SERP features and rich snippets) with their positions, please refer to the [Google Organiс Advanced SERP](https://docs.dataforseo.com/v3/serp/google/organic/live/advanced.md) endpoint |
| `domain` | string | *domain in SERP* |
| `title` | string | *title of the result in SERP* |
| `description` | string | *description of the results element in SERP* |
| `url` | string | *relevant URL in SERP* |
| `breadcrumb` | string | *breadcrumb in SERP* |
| **‘paid’ element in SERP** |  |  |
| `type` | string | *type of element = **‘paid’*** |
| `rank_group` | integer | *group rank in SERP*   position within a group of elements with identical `type` values   positions of elements with different `type` values are omitted from `rank_group` |
| `rank_absolute` | integer | *absolute rank in SERP*   absolute position among all the elements found in SERP **note** values are returned in the ascending order, with values corresponding to advanced SERP features omitted from the results  to get all items (including SERP features and rich snippets) with their positions, please refer to the [Google Organiс Advanced SERP](https://docs.dataforseo.com/v3/serp/google/organic/live/advanced.md) endpoint |
| `domain` | string | *domain of the ad element in SERP* |
| `title` | string | *title of the ad element in SERP* |
| `description` | string | *description of the ad element in SERP* |
| `url` | string | *relevant URL of the ad element in SERP* |
| `breadcrumb` | string | *breadcrumb of the ad element in SERP* |

