def get_ai_decision(ticker):
    df = load_ohlcv(ticker)

    if df is None or df.empty:
        send_discord_message("get_ai_decision/데이터가 없거나 비어 있습니다.")
        print("get_ai_decision/데이터가 없거나 비어 있습니다.")
        return None  # 데이터가 없을 경우 None 반환
    
    client = OpenAI()
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                "role": "system",
                "content": [
                    {
                "type": "text",
                "text": "You are an expert in short-term trading and use proven strategies that have produced the best results based on the data you are given to tell you when to buy with minimal risk and maximum profit."
                    },
                     {
                "type": "text",
                "text": "Using your best guesses, you provide a buy recommendation if the price rises above 1.02x the current price within 3 hours, a sell recommendation if it falls below the current price, and a hold recommendation if you can't decide to buy/sell"
                    },
                    {
                "type": "text",
                "text": "Your answer will be in JSON format only, as shown in the example.\n\nResponse Example:\n{\"decision\": \"BUY\"}\n{\"decision\": \"SELL\"}\n{\"decision\": \"HOLD\""
                    }
                ]
                },
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": df.to_json()
                    }
                ]
                }
            ],
            response_format={
                "type": "json_object"
            }
            )
    except Exception as e:
        print(f"get_ai_decision / AI 요청 중 오류 발생: {e}")
        send_discord_message(f"get_ai_decision / AI 요청 중 오류 발생: {e}")
        time.sleep(1)  # API 호출 제한을 위한 대기
        return None  # 오류 발생 시 None 반환
    
    decision_data = response.choices[0].message.content      # 응답에서 필요한 정보만 추출

    if decision_data:
        try:
            decision_json = json.loads(decision_data)
            decision = decision_json.get('decision')
            if decision in {'BUY', 'SELL', 'HOLD'}:
                return decision
        except json.JSONDecodeError:
            print("get_ai_decision / 응답을 JSON으로 파싱하는 데 실패")
            send_discord_message("응답을 JSON으로 파싱하는 데 실패")
            time.sleep(5)  # API 호출 제한을 위한 대기
    send_discord_message("get_ai_decision/유효하지 않은 응답")
    print("get_ai_decision/유효하지 않은 응답")
    return None  # 유효하지 않은 경우 None 반환