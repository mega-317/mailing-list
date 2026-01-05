def final_cfp_node(state) -> dict:
    
    conf_name = state.get("selected_conf_name")
    start_date = state.get("start_date")
    sub_deadline = state.get("sub_deadline")
    conf_url = state.get("conf_url")
    
    final_cfp = bool(conf_name and sub_deadline)
    
    conf_name_final = conf_name if final_cfp else None
    start_date_final = start_date if final_cfp else None
    sub_deadline_final = sub_deadline if final_cfp else None
    conf_url_final = conf_url if final_cfp else None
    
    print("최종 CFP 여부 결정 완료")
        
    return {
        "is_cfp_final": final_cfp,
        "conf_name_final": conf_name_final,
        "start_date_final": start_date_final,
        "sub_deadline_final": sub_deadline_final,
        "conf_url_final": conf_url_final
    }