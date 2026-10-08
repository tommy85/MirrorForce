-- Tierra, Source of Destruction
-- MirrorForce runtime-equivalent performance override.
-- The upstream CheckSubGroup(10 of N) enumerates combinations and becomes
-- effectively non-terminating with large hands.  This version checks distinct
-- names and zone release in polynomial time, then offers only choices that
-- still admit a valid completion.
function c91588074.initial_effect(c)
	c:EnableReviveLimit()
	local e1=Effect.CreateEffect(c)
	e1:SetType(EFFECT_TYPE_FIELD)
	e1:SetCode(EFFECT_SPSUMMON_PROC)
	e1:SetProperty(EFFECT_FLAG_CANNOT_DISABLE+EFFECT_FLAG_UNCOPYABLE)
	e1:SetRange(LOCATION_HAND)
	e1:SetCondition(c91588074.spcon)
	e1:SetTarget(c91588074.sptg)
	e1:SetOperation(c91588074.spop)
	c:RegisterEffect(e1)
	local e2=Effect.CreateEffect(c)
	e2:SetType(EFFECT_TYPE_SINGLE)
	e2:SetCode(EFFECT_CANNOT_DISABLE_SPSUMMON)
	e2:SetProperty(EFFECT_FLAG_CANNOT_DISABLE+EFFECT_FLAG_UNCOPYABLE)
	c:RegisterEffect(e2)
	local e3=Effect.CreateEffect(c)
	e3:SetProperty(EFFECT_FLAG_CANNOT_DISABLE+EFFECT_FLAG_UNCOPYABLE)
	e3:SetType(EFFECT_TYPE_SINGLE)
	e3:SetCode(EFFECT_SPSUMMON_CONDITION)
	c:RegisterEffect(e3)
	local e4=Effect.CreateEffect(c)
	e4:SetDescription(aux.Stringid(91588074,0))
	e4:SetCategory(CATEGORY_TODECK)
	e4:SetType(EFFECT_TYPE_TRIGGER_F+EFFECT_TYPE_SINGLE)
	e4:SetCode(EVENT_SPSUMMON_SUCCESS)
	e4:SetTarget(c91588074.tdtg)
	e4:SetOperation(c91588074.tdop)
	c:RegisterEffect(e4)
end

function c91588074.notselectedname(c,sg)
	return not sg:IsExists(Card.IsCode,1,nil,c:GetCode())
end

function c91588074.freeszone(c,sg,tp)
	local tg=sg:Clone()
	tg:AddCard(c)
	return Duel.GetMZoneCount(tp,tg)>0
end

function c91588074.hasvalidgroup(g,tp)
	if g:GetClassCount(Card.GetCode)<10 then return false end
	if Duel.GetMZoneCount(tp)>0 then return true end
	local mg=g:Filter(Card.IsLocation,nil,LOCATION_MZONE)
	for mc in aux.Next(mg) do
		local sg=Group.FromCards(mc)
		if Duel.GetMZoneCount(tp,sg)>0
			and g:Filter(c91588074.notselectedname,nil,sg):GetClassCount(Card.GetCode)>=9 then
			return true
		end
	end
	return false
end

function c91588074.pickfilter(c,sg,g,tp,left)
	if not c91588074.notselectedname(c,sg) then return false end
	local tg=sg:Clone()
	tg:AddCard(c)
	local rg=g:Filter(c91588074.notselectedname,nil,tg)
	if rg:GetClassCount(Card.GetCode)<left then return false end
	if Duel.GetMZoneCount(tp,tg)>0 then return true end
	return left>0 and rg:IsExists(c91588074.freeszone,1,nil,tg,tp)
end

function c91588074.spcon(e,c)
	if c==nil then return true end
	local tp=c:GetControler()
	local g=Duel.GetMatchingGroup(Card.IsAbleToDeckOrExtraAsCost,tp,
		LOCATION_HAND+LOCATION_ONFIELD,0,c)
	return c91588074.hasvalidgroup(g,tp)
end

function c91588074.sptg(e,tp,eg,ep,ev,re,r,rp,chk,c)
	local g=Duel.GetMatchingGroup(Card.IsAbleToDeckOrExtraAsCost,tp,
		LOCATION_HAND+LOCATION_ONFIELD,0,c)
	local sg=Group.CreateGroup()
	for index=1,10 do
		local left=10-index
		local cg=g:Filter(c91588074.pickfilter,nil,sg,g,tp,left)
		if #cg==0 then return false end
		Duel.Hint(HINT_SELECTMSG,tp,HINTMSG_TODECK)
		local tc=cg:SelectUnselect(nil,tp,false,true,1,1)
		if not tc then return false end
		sg:AddCard(tc)
	end
	if #sg~=10 or not aux.dncheck(sg) or not aux.mzctcheck(sg,tp) then
		return false
	end
	sg:KeepAlive()
	e:SetLabelObject(sg)
	return true
end

function c91588074.spop(e,tp,eg,ep,ev,re,r,rp,c)
	local g=e:GetLabelObject()
	local cg=g:Filter(Card.IsFacedown,nil)
	if cg:GetCount()>0 then Duel.ConfirmCards(1-tp,cg) end
	Duel.SendtoDeck(g,nil,SEQ_DECKSHUFFLE,REASON_SPSUMMON)
	g:DeleteGroup()
end

function c91588074.tdfilter(c)
	return (c:IsLocation(0x1e) or (c:IsFaceup() and c:IsType(TYPE_PENDULUM)))
		and c:IsAbleToDeck()
end
function c91588074.tdtg(e,tp,eg,ep,ev,re,r,rp,chk)
	if chk==0 then return true end
	local g=Duel.GetMatchingGroup(c91588074.tdfilter,tp,0x5e,0x5e,e:GetHandler())
	Duel.SetOperationInfo(0,CATEGORY_TODECK,g,g:GetCount(),0,0x5e)
	Duel.SetChainLimit(aux.FALSE)
end
function c91588074.tdop(e,tp,eg,ep,ev,re,r,rp)
	local g=Duel.GetMatchingGroup(c91588074.tdfilter,tp,0x5e,0x5e,aux.ExceptThisCard(e))
	if aux.NecroValleyNegateCheck(g) then return end
	Duel.SendtoDeck(g,nil,SEQ_DECKSHUFFLE,REASON_EFFECT)
end
